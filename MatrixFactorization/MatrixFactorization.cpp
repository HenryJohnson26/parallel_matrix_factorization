#include "Benchmark.h"
#include "CCDPP.h"
#include "RatingLoader.h"
#include "SparseRatings.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace
{
    int parseIntOrDefault(char* value, int defaultValue)
    {
        if (value == nullptr)
        {
            return defaultValue;
        }

        return std::stoi(value);
    }

    double parseDoubleOrDefault(char* value, double defaultValue)
    {
        if (value == nullptr)
        {
            return defaultValue;
        }

        return std::stod(value);
    }

    void printUsage()
    {
        std::cout
            << "Usage:\n"
            << "\n"
            << "  Train normally:\n"
            << "    CCDPlusPlus.exe train <ratings-file> [max-ratings] [rank] [outer] [inner] [threads]\n"
            << "\n"
            << "  Strong scaling benchmark:\n"
            << "    CCDPlusPlus.exe bench-strong <ratings-file> [max-ratings] [rank] [outer] [inner] [max-threads]\n"
            << "\n"
            << "  Weak scaling benchmark:\n"
            << "    CCDPlusPlus.exe bench-weak <ratings-file> [ratings-per-thread] [rank] [outer] [inner] [max-threads]\n"
            << "\n"
            << "  Ablation benchmark:\n"
            << "    CCDPlusPlus.exe bench-ablation <ratings-file> [max-ratings] [rank] [outer] [inner] [threads]\n"
            << "\n"
            << "Examples:\n"
            << "    CCDPlusPlus.exe train C:\\\\data\\\\ml-1m\\\\ratings.dat 1000000 40 5 5 8\n"
            << "    CCDPlusPlus.exe bench-strong C:\\\\data\\\\ml-1m\\\\ratings.dat 1000000 40 3 5 8\n"
            << "    CCDPlusPlus.exe bench-weak C:\\\\data\\\\ml-1m\\\\ratings.dat 125000 40 3 5 8\n"
            << "    CCDPlusPlus.exe bench-ablation C:\\\\data\\\\ml-1m\\\\ratings.dat 1000000 40 3 5 8\n";
    }

    CCDPPConfig makeBaseConfig(
        int rank,
        int outerIterations,
        int innerIterations,
        int threads,
        bool verbose)
    {
        CCDPPConfig cfg;
        cfg.rank = rank;
        cfg.lambda = 0.1;
        cfg.outerIterations = outerIterations;
        cfg.innerIterations = innerIterations;
        cfg.useOpenMP = threads > 1;
        cfg.numThreads = threads;
        cfg.parallelVectorLoops = true;
        cfg.dynamicScheduling = true;
        cfg.dynamicChunkSize = 64;
        cfg.computeRmseEachOuter = verbose;
        cfg.verbose = verbose;

        return cfg;
    }

    SparseRatings makeSparseFromLoaded(const LoadedRatings& loaded)
    {
        int maxUser = -1;
        int maxItem = -1;

        for (const auto& rating : loaded.ratings)
        {
            maxUser = std::max(maxUser, std::get<0>(rating));
            maxItem = std::max(maxItem, std::get<1>(rating));
        }

        return SparseRatings(maxUser + 1, maxItem + 1, loaded.ratings);
    }
}

int main(int argc, char* argv[])
{
    try
    {
        if (argc < 2)
        {
            printUsage();
            return 0;
        }

        std::string command = argv[1];

        if (command == "train")
        {
            if (argc < 3)
            {
                printUsage();
                return 1;
            }

            std::string filename = argv[2];

            int maxRatings = argc >= 4 ? std::stoi(argv[3]) : -1;
            int rank = argc >= 5 ? std::stoi(argv[4]) : 2;
            int outerIterations = argc >= 6 ? std::stoi(argv[5]) : 10;
            int innerIterations = argc >= 7 ? std::stoi(argv[6]) : 5;
            int threads = argc >= 8 ? std::stoi(argv[7]) : 1;

            LoadedRatings loaded = loadRatingsFromFile(filename, false, maxRatings);
            SparseRatings data = makeSparseFromLoaded(loaded);

            CCDPPConfig cfg = makeBaseConfig(
                rank,
                outerIterations,
                innerIterations,
                threads,
                true);

            CCDPP model(cfg);
            model.fit(data);

            std::cout << "\nFirst few predictions for observed ratings:\n";

            int count = std::min<int>(20, static_cast<int>(data.entries.size()));

            for (int index = 0; index < count; ++index)
            {
                const ObservedEntry& entry = data.entries[index];

                std::cout << "user " << entry.user
                    << ", item " << entry.item
                    << ", actual " << entry.rating
                    << ", predicted " << model.predict(entry.user, entry.item)
                    << "\n";
            }

            return 0;
        }

        if (command == "bench-strong")
        {
            if (argc < 3)
            {
                printUsage();
                return 1;
            }

            std::string filename = argv[2];

            int maxRatings = argc >= 4 ? std::stoi(argv[3]) : 1000000;
            int rank = argc >= 5 ? std::stoi(argv[4]) : 40;
            int outerIterations = argc >= 6 ? std::stoi(argv[5]) : 3;
            int innerIterations = argc >= 7 ? std::stoi(argv[6]) : 5;
            int maxThreads = argc >= 8 ? std::stoi(argv[7]) : 8;

            LoadedRatings loaded = loadRatingsFromFile(filename, false, -1);

            CCDPPConfig baseConfig = makeBaseConfig(
                rank,
                outerIterations,
                innerIterations,
                1,
                false);

            std::vector<BenchmarkRow> rows = runStrongScalingBenchmark(
                loaded,
                maxRatings,
                baseConfig,
                maxThreads);

            writeBenchmarkCsv("strong_scaling.csv", rows);
            std::cout << "Wrote strong_scaling.csv\n";

            return 0;
        }

        if (command == "bench-weak")
        {
            if (argc < 3)
            {
                printUsage();
                return 1;
            }

            std::string filename = argv[2];

            int ratingsPerThread = argc >= 4 ? std::stoi(argv[3]) : 125000;
            int rank = argc >= 5 ? std::stoi(argv[4]) : 40;
            int outerIterations = argc >= 6 ? std::stoi(argv[5]) : 3;
            int innerIterations = argc >= 7 ? std::stoi(argv[6]) : 5;
            int maxThreads = argc >= 8 ? std::stoi(argv[7]) : 8;

            LoadedRatings loaded = loadRatingsFromFile(filename, false, -1);

            CCDPPConfig baseConfig = makeBaseConfig(
                rank,
                outerIterations,
                innerIterations,
                1,
                false);

            std::vector<BenchmarkRow> rows = runWeakScalingBenchmark(
                loaded,
                ratingsPerThread,
                baseConfig,
                maxThreads);

            writeBenchmarkCsv("weak_scaling.csv", rows);
            std::cout << "Wrote weak_scaling.csv\n";

            return 0;
        }

        if (command == "bench-ablation")
        {
            if (argc < 3)
            {
                printUsage();
                return 1;
            }

            std::string filename = argv[2];

            int maxRatings = argc >= 4 ? std::stoi(argv[3]) : 1000000;
            int rank = argc >= 5 ? std::stoi(argv[4]) : 40;
            int outerIterations = argc >= 6 ? std::stoi(argv[5]) : 3;
            int innerIterations = argc >= 7 ? std::stoi(argv[6]) : 5;
            int threads = argc >= 8 ? std::stoi(argv[7]) : 8;

            LoadedRatings loaded = loadRatingsFromFile(filename, false, -1);

            CCDPPConfig baseConfig = makeBaseConfig(
                rank,
                outerIterations,
                innerIterations,
                threads,
                false);

            std::vector<BenchmarkRow> rows = runAblationBenchmark(
                loaded,
                maxRatings,
                baseConfig,
                threads);

            writeBenchmarkCsv("ablation.csv", rows);
            std::cout << "Wrote ablation.csv\n";

            return 0;
        }

        // Backward-compatible behavior:
        // If the first argument is not a recognized command, treat it as a ratings file.
        {
            std::string filename = argv[1];
            int maxRatings = argc >= 3 ? std::stoi(argv[2]) : -1;

            LoadedRatings loaded = loadRatingsFromFile(filename, false, maxRatings);
            SparseRatings data = makeSparseFromLoaded(loaded);

            CCDPPConfig cfg = makeBaseConfig(
                2,
                10,
                5,
                1,
                true);

            CCDPP model(cfg);
            model.fit(data);
        }
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}