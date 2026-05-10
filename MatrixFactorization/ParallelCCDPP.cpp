#include "ParallelCCDPP.h"

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <omp.h>

#define UCHUNK_SIZE 4
#define VCHUNK_SIZE 4


ParallelCCDPP::ParallelCCDPP(
    int rank,
    double lambdaValue,
    int innerIterationCount,
    int outerIterationCount)
    : k(rank),
    lambda(lambdaValue),
    innerIterations(innerIterationCount),
    outerIterations(outerIterationCount)
{
    if (k <= 0)
    {
        throw std::runtime_error("Rank k must be positive.");
    }

    if (lambda <= 0.0)
    {
        throw std::runtime_error("Lambda must be positive.");
    }

    if (innerIterations <= 0)
    {
        throw std::runtime_error("Inner iterations must be positive.");
    }

    if (outerIterations <= 0)
    {
        throw std::runtime_error("Outer iterations must be positive.");
    }
}

void ParallelCCDPP::fit(SparseRatings& data)
{
    initializeFactors(data.numUsers, data.numItems);

    std::cout << "Users: " << data.numUsers << "\n";
    std::cout << "Items: " << data.numItems << "\n";
    std::cout << "Observed ratings: " << data.nonzeroCount() << "\n\n";

    std::cout << "Initial training RMSE: " << trainingRmse(data) << "\n";

    for (int outer = 0; outer < outerIterations; ++outer)
    {
        for (int t = 0; t < k; ++t)
        {
            updateOneFeature(data, t);
        }

        std::cout << "After outer iteration " << (outer + 1)
            << ", training RMSE: " << trainingRmse(data) << "\n";
    }
}

double ParallelCCDPP::predict(int user, int item) const
{
    double result = 0.0;

    for (int t = 0; t < k; ++t)
    {
        result += W[user][t] * H[item][t];
    }

    return result;
}

double ParallelCCDPP::trainingRmse(const SparseRatings& data) const
{
    double sumSquaredError = 0.0;

    for (const ObservedEntry& entry : data.entries)
    {
        sumSquaredError += entry.residual * entry.residual;
    }

    return std::sqrt(sumSquaredError / static_cast<double>(data.nonzeroCount()));
}

void ParallelCCDPP::initializeFactors(int numUsers, int numItems)
{
    W.assign(numUsers, std::vector<double>(k, 0.0));
    H.assign(numItems, std::vector<double>(k, 0.0));

    // If both W and H are initialized to zero, the updates remain stuck at zero.
    // We keep W at zero and initialize H to small nonzero values.
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(0.01, 0.10);

    for (int item = 0; item < numItems; ++item)
    {
        for (int t = 0; t < k; ++t)
        {
            H[item][t] = dist(rng);
        }
    }
}

void ParallelCCDPP::updateOneFeature(SparseRatings& data, int t)
{
    std::vector<double> u(data.numUsers, 0.0);
    std::vector<double> v(data.numItems, 0.0);

    for (int user = 0; user < data.numUsers; ++user)
    {
        u[user] = W[user][t];
    }

    for (int item = 0; item < data.numItems; ++item)
    {
        v[item] = H[item][t];
    }

    // Construct R_hat = R + W[:, t] * H[:, t]^T.
    #pragma omp parallel for schedule(dynamic)
    // for (ObservedEntry& entry : data.entries)
    for (int i = 0; i < (int)data.entries.size(); i++)
    {
        ObservedEntry& entry = data.entries[i];
        entry.residual += u[entry.user] * v[entry.item];
    }

    // Inner CCD iterations for the rank-one problem.
    for (int inner = 0; inner < innerIterations; ++inner)
    {
        updateU(data, u, v);
        updateV(data, u, v);
    }

    // Store the updated feature column.
    #pragma omp parallel for schedule(dynamic)
    for (int user = 0; user < data.numUsers; ++user)
    {
        W[user][t] = u[user];
    }

    #pragma omp parallel for schedule(dynamic)
    for (int item = 0; item < data.numItems; ++item)
    {
        H[item][t] = v[item];
    }

    // Convert R_hat back into the actual residual R.
    #pragma omp parallel for schedule(dynamic)
    // for (ObservedEntry& entry : data.entries)
    for (int i = 0; i < (int)data.entries.size(); i++)
    {
        ObservedEntry& entry = data.entries[i];
        entry.residual -= u[entry.user] * v[entry.item];
    }
}

void ParallelCCDPP::updateU(
    const SparseRatings& data,
    std::vector<double>& u,
    const std::vector<double>& v)
{
        #pragma omp parallel for schedule(dynamic, UCHUNK_SIZE)
    for (int user = 0; user < data.numUsers; ++user)
    {
        double numerator = 0.0;
        double denominator = lambda;

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

void ParallelCCDPP::updateV(
    const SparseRatings& data,
    const std::vector<double>& u,
    std::vector<double>& v)
{
    #pragma omp parallel for schedule(dynamic, VCHUNK_SIZE)
    for (int item = 0; item < data.numItems; ++item)
    {
        double numerator = 0.0;
        double denominator = lambda;

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
