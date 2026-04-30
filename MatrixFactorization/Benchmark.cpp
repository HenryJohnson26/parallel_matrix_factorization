#include "Benchmark.h"

#include "SparseRatings.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace
{
    std::vector<int> makeThreadList(int maxThreads)
    {
        std::vector<int> threads;

        for (int t = 1; t <= maxThreads; t *= 2)
        {
            threads.push_back(t);
        }

        if (threads.empty() || threads.back() != maxThreads)
        {
            threads.push_back(maxThreads);
        }

        return threads;
    }

    int clampRatingCount(const LoadedRatings& loaded, int requestedRatings)
    {
        if (requestedRatings <= 0)
        {
            return static_cast<int>(loaded.ratings.size());
        }

        return std::min<int>(
            requestedRatings,
            static_cast<int>(loaded.ratings.size()));
    }

    SparseRatings makeSparsePrefix(const LoadedRatings& loaded, int requestedRatings)
    {
        int actualRatings = clampRatingCount(loaded, requestedRatings);

        std::vector<std::tuple<int, int, double>> ratings;
        ratings.reserve(actualRatings);

        int maxUser = -1;
        int maxItem = -1;

        for (int i = 0; i < actualRatings; ++i)
        {
            const auto& rating = loaded.ratings[static_cast<std::size_t>(i)];

            int user = std::get<0>(rating);
            int item = std::get<1>(rating);

            maxUser = std::max(maxUser, user);
            maxItem = std::max(maxItem, item);

            ratings.push_back(rating);
        }

        int numUsers = maxUser + 1;
        int numItems = maxItem + 1;

        return SparseRatings(numUsers, numItems, ratings);
    }

    BenchmarkRow runOneCase(
        const std::string& experiment,
        const std::string& mode,
        const LoadedRatings& loaded,
        int requestedRatings,
        CCDPPConfig config)
    {
        SparseRatings data = makeSparsePrefix(loaded, requestedRatings);

        config.verbose = false;

        CCDPP model(config);
        FitResult result = model.fit(data);

        BenchmarkRow row;
        row.experiment = experiment;
        row.mode = mode;
        row.threads = config.useOpenMP ? config.numThreads : 1;
        row.requestedRatings = requestedRatings;
        row.actualRatings = static_cast<int>(data.nonzeroCount());
        row.users = data.numUsers;
        row.items = data.numItems;
        row.seconds = result.elapsedSeconds;
        row.finalRmse = result.finalRmse;

        return row;
    }

    CCDPPConfig makeSerialConfig(const CCDPPConfig& base)
    {
        CCDPPConfig cfg = base;
        cfg.useOpenMP = false;
        cfg.numThreads = 1;
        cfg.parallelVectorLoops = false;
        cfg.dynamicScheduling = false;
        cfg.computeRmseEachOuter = false;
        return cfg;
    }

    CCDPPConfig makeOmpUvStaticConfig(const CCDPPConfig& base, int threads)
    {
        CCDPPConfig cfg = base;
        cfg.useOpenMP = true;
        cfg.numThreads = threads;

        // Only updateU/updateV are parallel.
        // R_hat and residual loops remain serial.
        cfg.parallelVectorLoops = false;

        // Static scheduling means each thread receives a fixed block.
        cfg.dynamicScheduling = false;

        cfg.computeRmseEachOuter = false;
        return cfg;
    }

    CCDPPConfig makeOmpAllStaticConfig(const CCDPPConfig& base, int threads)
    {
        CCDPPConfig cfg = base;
        cfg.useOpenMP = true;
        cfg.numThreads = threads;

        // Parallelize R_hat construction, residual update, and feature copy loops.
        cfg.parallelVectorLoops = true;

        // Still use static scheduling for updateU/updateV.
        cfg.dynamicScheduling = false;

        cfg.computeRmseEachOuter = false;
        return cfg;
    }

    CCDPPConfig makeOmpAllDynamicConfig(const CCDPPConfig& base, int threads)
    {
        CCDPPConfig cfg = base;
        cfg.useOpenMP = true;
        cfg.numThreads = threads;

        // Parallelize all safe independent loops.
        cfg.parallelVectorLoops = true;

        // Use dynamic scheduling for updateU/updateV.
        cfg.dynamicScheduling = true;
        cfg.dynamicChunkSize = 64;

        cfg.computeRmseEachOuter = false;
        return cfg;
    }

    void fillSpeedupAndEfficiency(std::vector<BenchmarkRow>& rows)
    {
        if (rows.empty())
        {
            return;
        }

        double baselineSeconds = rows.front().seconds;

        for (BenchmarkRow& row : rows)
        {
            row.speedup = baselineSeconds / row.seconds;
            row.efficiency = row.speedup / static_cast<double>(row.threads);
        }
    }
}

std::vector<BenchmarkRow> runStrongScalingBenchmark(
    const LoadedRatings& loaded,
    int maxRatings,
    const CCDPPConfig& baseConfig,
    int maxThreads)
{
    std::vector<BenchmarkRow> rows;

    std::vector<int> threadCounts = makeThreadList(maxThreads);

    for (int threads : threadCounts)
    {
        CCDPPConfig cfg = makeOmpAllDynamicConfig(baseConfig, threads);

        BenchmarkRow row = runOneCase(
            "strong_scaling",
            "omp_all_dynamic",
            loaded,
            maxRatings,
            cfg);

        rows.push_back(row);

        std::cout << "[strong] threads=" << threads
            << ", ratings=" << row.actualRatings
            << ", seconds=" << row.seconds
            << ", rmse=" << row.finalRmse << "\n";
    }

    fillSpeedupAndEfficiency(rows);
    return rows;
}

std::vector<BenchmarkRow> runWeakScalingBenchmark(
    const LoadedRatings& loaded,
    int ratingsPerThread,
    const CCDPPConfig& baseConfig,
    int maxThreads)
{
    std::vector<BenchmarkRow> rows;

    std::vector<int> threadCounts = makeThreadList(maxThreads);

    for (int threads : threadCounts)
    {
        int requestedRatings = ratingsPerThread * threads;

        CCDPPConfig cfg = makeOmpAllDynamicConfig(baseConfig, threads);

        BenchmarkRow row = runOneCase(
            "weak_scaling",
            "omp_all_dynamic",
            loaded,
            requestedRatings,
            cfg);

        rows.push_back(row);

        std::cout << "[weak] threads=" << threads
            << ", ratings=" << row.actualRatings
            << ", seconds=" << row.seconds
            << ", rmse=" << row.finalRmse << "\n";
    }

    // For weak scaling, speedup is less meaningful than time stability.
    // Still fill it for plotting consistency.
    fillSpeedupAndEfficiency(rows);
    return rows;
}

std::vector<BenchmarkRow> runAblationBenchmark(
    const LoadedRatings& loaded,
    int maxRatings,
    const CCDPPConfig& baseConfig,
    int threads)
{
    std::vector<BenchmarkRow> rows;

    std::vector<std::pair<std::string, CCDPPConfig>> cases;

    cases.push_back({
        "serial_flat_factors",
        makeSerialConfig(baseConfig)
        });

    cases.push_back({
        "omp_uv_static",
        makeOmpUvStaticConfig(baseConfig, threads)
        });

    cases.push_back({
        "omp_all_static",
        makeOmpAllStaticConfig(baseConfig, threads)
        });

    cases.push_back({
        "omp_all_dynamic",
        makeOmpAllDynamicConfig(baseConfig, threads)
        });

    for (const auto& item : cases)
    {
        const std::string& mode = item.first;
        CCDPPConfig cfg = item.second;

        BenchmarkRow row = runOneCase(
            "ablation",
            mode,
            loaded,
            maxRatings,
            cfg);

        rows.push_back(row);

        std::cout << "[ablation] mode=" << mode
            << ", threads=" << row.threads
            << ", ratings=" << row.actualRatings
            << ", seconds=" << row.seconds
            << ", rmse=" << row.finalRmse << "\n";
    }

    fillSpeedupAndEfficiency(rows);
    return rows;
}

void writeBenchmarkCsv(
    const std::string& filename,
    const std::vector<BenchmarkRow>& rows)
{
    std::ofstream output(filename);

    if (!output)
    {
        throw std::runtime_error("Could not open benchmark CSV for writing: " + filename);
    }

    output
        << "experiment,"
        << "mode,"
        << "threads,"
        << "requested_ratings,"
        << "actual_ratings,"
        << "users,"
        << "items,"
        << "seconds,"
        << "speedup,"
        << "efficiency,"
        << "final_rmse\n";

    for (const BenchmarkRow& row : rows)
    {
        output
            << row.experiment << ","
            << row.mode << ","
            << row.threads << ","
            << row.requestedRatings << ","
            << row.actualRatings << ","
            << row.users << ","
            << row.items << ","
            << row.seconds << ","
            << row.speedup << ","
            << row.efficiency << ","
            << row.finalRmse << "\n";
    }
}