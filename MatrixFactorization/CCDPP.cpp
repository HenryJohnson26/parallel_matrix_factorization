#include "CCDPP.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

CCDPP::CCDPP(const CCDPPConfig& cfg)
    : config(cfg),
    k(cfg.rank)
{
    if (config.rank <= 0)
    {
        throw std::runtime_error("Rank k must be positive.");
    }

    if (config.lambda <= 0.0)
    {
        throw std::runtime_error("Lambda must be positive.");
    }

    if (config.innerIterations <= 0)
    {
        throw std::runtime_error("Inner iterations must be positive.");
    }

    if (config.outerIterations <= 0)
    {
        throw std::runtime_error("Outer iterations must be positive.");
    }

    if (config.numThreads <= 0)
    {
        throw std::runtime_error("Number of threads must be positive.");
    }

#ifndef _OPENMP
    if (config.useOpenMP)
    {
        std::cerr << "Warning: OpenMP was requested, but this file was not compiled with OpenMP.\n";
    }
#endif
}

double& CCDPP::WAt(int user, int feature)
{
    return W[static_cast<std::size_t>(user) * k + feature];
}

const double& CCDPP::WAt(int user, int feature) const
{
    return W[static_cast<std::size_t>(user) * k + feature];
}

double& CCDPP::HAt(int item, int feature)
{
    return H[static_cast<std::size_t>(item) * k + feature];
}

const double& CCDPP::HAt(int item, int feature) const
{
    return H[static_cast<std::size_t>(item) * k + feature];
}

FitResult CCDPP::fit(SparseRatings& data)
{
#ifdef _OPENMP
    if (config.useOpenMP)
    {
        omp_set_num_threads(config.numThreads);
    }
#endif

    initializeFactors(data.numUsers, data.numItems);

    FitResult result;
    result.initialRmse = trainingRmse(data);

    if (config.verbose)
    {
        std::cout << "Users: " << data.numUsers << "\n";
        std::cout << "Items: " << data.numItems << "\n";
        std::cout << "Observed ratings: " << data.nonzeroCount() << "\n";
        std::cout << "OpenMP enabled: " << (config.useOpenMP ? "yes" : "no") << "\n";
        std::cout << "Threads: " << config.numThreads << "\n";
        std::cout << "Initial training RMSE: " << result.initialRmse << "\n";
    }

    // Optimization #2:
    // Reuse u and v workspaces instead of allocating them inside every feature update.
    //
    // CCD++ updates one latent feature at a time:
    //   W[:, t], H[:, t]
    //
    // For each feature, we need temporary rank-one vectors u and v.
    // Allocating them once avoids repeated heap allocation overhead.
    std::vector<double> u(data.numUsers, 0.0);
    std::vector<double> v(data.numItems, 0.0);

    auto startTime = std::chrono::high_resolution_clock::now();

    for (int outer = 0; outer < config.outerIterations; ++outer)
    {
        for (int feature = 0; feature < k; ++feature)
        {
            updateOneFeature(data, feature, u, v);
        }

        if (config.computeRmseEachOuter)
        {
            double rmse = trainingRmse(data);
            result.rmseByOuterIteration.push_back(rmse);

            if (config.verbose)
            {
                std::cout << "After outer iteration " << (outer + 1)
                    << ", training RMSE: " << rmse << "\n";
            }
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;

    result.elapsedSeconds = elapsed.count();
    result.finalRmse = trainingRmse(data);

    if (config.verbose)
    {
        std::cout << "Final training RMSE: " << result.finalRmse << "\n";
        std::cout << "Training time, excluding initial/final RMSE scans: "
            << result.elapsedSeconds << " seconds\n";
    }

    return result;
}

double CCDPP::predict(int user, int item) const
{
    double result = 0.0;

    for (int feature = 0; feature < k; ++feature)
    {
        result += WAt(user, feature) * HAt(item, feature);
    }

    return result;
}

double CCDPP::trainingRmse(const SparseRatings& data) const
{
    double sumSquaredError = 0.0;

    // This loop is read-only and independent per rating.
    // It is safe to parallelize with a reduction.
    if (config.useOpenMP)
    {
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sumSquaredError) schedule(static)
        for (long long index = 0; index < static_cast<long long>(data.entries.size()); ++index)
        {
            const ObservedEntry& entry = data.entries[static_cast<std::size_t>(index)];
            sumSquaredError += entry.residual * entry.residual;
        }
#else
        for (const ObservedEntry& entry : data.entries)
        {
            sumSquaredError += entry.residual * entry.residual;
        }
#endif
    }
    else
    {
        for (const ObservedEntry& entry : data.entries)
        {
            sumSquaredError += entry.residual * entry.residual;
        }
    }

    return std::sqrt(sumSquaredError / static_cast<double>(data.nonzeroCount()));
}

void CCDPP::initializeFactors(int numUsers, int numItems)
{
    W.assign(static_cast<std::size_t>(numUsers) * k, 0.0);
    H.assign(static_cast<std::size_t>(numItems) * k, 0.0);

    // Same deterministic initialization as before.
    // W starts at zero, while H starts with small nonzero values.
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(0.01, 0.10);

    for (int item = 0; item < numItems; ++item)
    {
        for (int feature = 0; feature < k; ++feature)
        {
            HAt(item, feature) = dist(rng);
        }
    }
}

void CCDPP::updateOneFeature(
    SparseRatings& data,
    int feature,
    std::vector<double>& u,
    std::vector<double>& v)
{
    // ---------------------------------------------------------------------
    // Step A: Copy the current feature column into temporary vectors.
    //
    // Algorithmically:
    //   u = W[:, feature]
    //   v = H[:, feature]
    //
    // These loops are independent because each iteration writes a different
    // u[user] or v[item].
    // ---------------------------------------------------------------------

    if (config.useOpenMP && config.parallelVectorLoops)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
        for (int user = 0; user < data.numUsers; ++user)
        {
            u[user] = WAt(user, feature);
        }

#pragma omp parallel for schedule(static)
        for (int item = 0; item < data.numItems; ++item)
        {
            v[item] = HAt(item, feature);
        }
#else
        for (int user = 0; user < data.numUsers; ++user)
        {
            u[user] = WAt(user, feature);
        }

        for (int item = 0; item < data.numItems; ++item)
        {
            v[item] = HAt(item, feature);
        }
#endif
    }
    else
    {
        for (int user = 0; user < data.numUsers; ++user)
        {
            u[user] = WAt(user, feature);
        }

        for (int item = 0; item < data.numItems; ++item)
        {
            v[item] = HAt(item, feature);
        }
    }

    // ---------------------------------------------------------------------
    // Step B: Construct R_hat.
    //
    // Paper equation:
    //   R_hat_ij = R_ij + W_it * H_jt
    //
    // Our code temporarily stores R_hat inside entry.residual.
    //
    // Parallel safety:
    //   Each iteration updates a different entry.residual.
    //   No two threads write the same residual element.
    // ---------------------------------------------------------------------

    if (config.useOpenMP && config.parallelVectorLoops)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
        for (long long index = 0; index < static_cast<long long>(data.entries.size()); ++index)
        {
            ObservedEntry& entry = data.entries[static_cast<std::size_t>(index)];
            entry.residual += u[entry.user] * v[entry.item];
        }
#else
        for (ObservedEntry& entry : data.entries)
        {
            entry.residual += u[entry.user] * v[entry.item];
        }
#endif
    }
    else
    {
        for (ObservedEntry& entry : data.entries)
        {
            entry.residual += u[entry.user] * v[entry.item];
        }
    }

    // ---------------------------------------------------------------------
    // Step C: Solve the rank-one subproblem with inner CCD iterations.
    //
    // The paper alternates:
    //   update u using fixed v
    //   update v using fixed u
    //
    // Important synchronization detail:
    //   Each OpenMP parallel for has an implicit barrier at the end.
    //   That means all u values are complete before updateV begins,
    //   and all v values are complete before the next inner iteration.
    // ---------------------------------------------------------------------

    for (int inner = 0; inner < config.innerIterations; ++inner)
    {
        updateU(data, u, v);
        updateV(data, u, v);
    }

    // ---------------------------------------------------------------------
    // Step D: Store the updated feature column.
    //
    //   W[:, feature] = u
    //   H[:, feature] = v
    //
    // Parallel safety:
    //   Each iteration writes a different W or H element.
    // ---------------------------------------------------------------------

    if (config.useOpenMP && config.parallelVectorLoops)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
        for (int user = 0; user < data.numUsers; ++user)
        {
            WAt(user, feature) = u[user];
        }

#pragma omp parallel for schedule(static)
        for (int item = 0; item < data.numItems; ++item)
        {
            HAt(item, feature) = v[item];
        }
#else
        for (int user = 0; user < data.numUsers; ++user)
        {
            WAt(user, feature) = u[user];
        }

        for (int item = 0; item < data.numItems; ++item)
        {
            HAt(item, feature) = v[item];
        }
#endif
    }
    else
    {
        for (int user = 0; user < data.numUsers; ++user)
        {
            WAt(user, feature) = u[user];
        }

        for (int item = 0; item < data.numItems; ++item)
        {
            HAt(item, feature) = v[item];
        }
    }

    // ---------------------------------------------------------------------
    // Step E: Convert R_hat back into the true residual R.
    //
    // Paper equation:
    //   R_ij = R_hat_ij - u_i * v_j
    //
    // Parallel safety:
    //   Each iteration writes a different entry.residual.
    // ---------------------------------------------------------------------

    if (config.useOpenMP && config.parallelVectorLoops)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
        for (long long index = 0; index < static_cast<long long>(data.entries.size()); ++index)
        {
            ObservedEntry& entry = data.entries[static_cast<std::size_t>(index)];
            entry.residual -= u[entry.user] * v[entry.item];
        }
#else
        for (ObservedEntry& entry : data.entries)
        {
            entry.residual -= u[entry.user] * v[entry.item];
        }
#endif
    }
    else
    {
        for (ObservedEntry& entry : data.entries)
        {
            entry.residual -= u[entry.user] * v[entry.item];
        }
    }
}

void CCDPP::updateU(
    const SparseRatings& data,
    std::vector<double>& u,
    const std::vector<double>& v)
{
    // ---------------------------------------------------------------------
    // Update rule:
    //
    //   u_i = sum_j R_hat_ij * v_j / (lambda + sum_j v_j^2)
    //
    // Parallel safety:
    //   Each user update writes only u[user].
    //   The residuals and v vector are read-only in this phase.
    //
    // Performance note:
    //   Users may have very different numbers of ratings.
    //   Dynamic scheduling can reduce load imbalance.
    // ---------------------------------------------------------------------

    if (!config.useOpenMP)
    {
        for (int user = 0; user < data.numUsers; ++user)
        {
            double numerator = 0.0;
            double denominator = config.lambda;

            for (int entryIndex : data.entriesByUser[user])
            {
                const ObservedEntry& entry = data.entries[entryIndex];
                double vj = v[entry.item];

                numerator += entry.residual * vj;
                denominator += vj * vj;
            }

            u[user] = numerator / denominator;
        }

        return;
    }

#ifdef _OPENMP
    if (config.dynamicScheduling)
    {
        const int chunk = config.dynamicChunkSize;

#pragma omp parallel for schedule(dynamic, chunk)
        for (int user = 0; user < data.numUsers; ++user)
        {
            double numerator = 0.0;
            double denominator = config.lambda;

            for (int entryIndex : data.entriesByUser[user])
            {
                const ObservedEntry& entry = data.entries[entryIndex];
                double vj = v[entry.item];

                numerator += entry.residual * vj;
                denominator += vj * vj;
            }

            u[user] = numerator / denominator;
        }
    }
    else
    {
#pragma omp parallel for schedule(static)
        for (int user = 0; user < data.numUsers; ++user)
        {
            double numerator = 0.0;
            double denominator = config.lambda;

            for (int entryIndex : data.entriesByUser[user])
            {
                const ObservedEntry& entry = data.entries[entryIndex];
                double vj = v[entry.item];

                numerator += entry.residual * vj;
                denominator += vj * vj;
            }

            u[user] = numerator / denominator;
        }
    }
#else
    for (int user = 0; user < data.numUsers; ++user)
    {
        double numerator = 0.0;
        double denominator = config.lambda;

        for (int entryIndex : data.entriesByUser[user])
        {
            const ObservedEntry& entry = data.entries[entryIndex];
            double vj = v[entry.item];

            numerator += entry.residual * vj;
            denominator += vj * vj;
        }

        u[user] = numerator / denominator;
    }
#endif
}

void CCDPP::updateV(
    const SparseRatings& data,
    const std::vector<double>& u,
    std::vector<double>& v)
{
    // ---------------------------------------------------------------------
    // Update rule:
    //
    //   v_j = sum_i R_hat_ij * u_i / (lambda + sum_i u_i^2)
    //
    // Parallel safety:
    //   Each item update writes only v[item].
    //   The residuals and u vector are read-only in this phase.
    //
    // Performance note:
    //   Items also have uneven numbers of ratings, especially popular movies.
    //   Dynamic scheduling helps keep threads busy.
    // ---------------------------------------------------------------------

    if (!config.useOpenMP)
    {
        for (int item = 0; item < data.numItems; ++item)
        {
            double numerator = 0.0;
            double denominator = config.lambda;

            for (int entryIndex : data.entriesByItem[item])
            {
                const ObservedEntry& entry = data.entries[entryIndex];
                double ui = u[entry.user];

                numerator += entry.residual * ui;
                denominator += ui * ui;
            }

            v[item] = numerator / denominator;
        }

        return;
    }

#ifdef _OPENMP
    if (config.dynamicScheduling)
    {
        const int chunk = config.dynamicChunkSize;

#pragma omp parallel for schedule(dynamic, chunk)
        for (int item = 0; item < data.numItems; ++item)
        {
            double numerator = 0.0;
            double denominator = config.lambda;

            for (int entryIndex : data.entriesByItem[item])
            {
                const ObservedEntry& entry = data.entries[entryIndex];
                double ui = u[entry.user];

                numerator += entry.residual * ui;
                denominator += ui * ui;
            }

            v[item] = numerator / denominator;
        }
    }
    else
    {
#pragma omp parallel for schedule(static)
        for (int item = 0; item < data.numItems; ++item)
        {
            double numerator = 0.0;
            double denominator = config.lambda;

            for (int entryIndex : data.entriesByItem[item])
            {
                const ObservedEntry& entry = data.entries[entryIndex];
                double ui = u[entry.user];

                numerator += entry.residual * ui;
                denominator += ui * ui;
            }

            v[item] = numerator / denominator;
        }
    }
#else
    for (int item = 0; item < data.numItems; ++item)
    {
        double numerator = 0.0;
        double denominator = config.lambda;

        for (int entryIndex : data.entriesByItem[item])
        {
            const ObservedEntry& entry = data.entries[entryIndex];
            double ui = u[entry.user];

            numerator += entry.residual * ui;
            denominator += ui * ui;
        }

        v[item] = numerator / denominator;
    }
#endif
}