#pragma once

#include "SparseRatings.h"

#include <string>
#include <vector>

struct CCDPPConfig
{
    int rank = 2;
    double lambda = 0.1;
    int innerIterations = 5;
    int outerIterations = 10;

    // Parallel control.
    bool useOpenMP = false;
    int numThreads = 1;

    // Optimization toggles for ablation studies.
    //
    // parallelVectorLoops:
    //   Parallelizes simple full-array loops such as:
    //     - copying W[:, t] into u
    //     - copying H[:, t] into v
    //     - constructing R_hat
    //     - updating the residual
    //
    // dynamicScheduling:
    //   Uses dynamic scheduling for updateU/updateV.
    //   This helps when users/items have very different numbers of ratings.
    bool parallelVectorLoops = true;
    bool dynamicScheduling = true;
    int dynamicChunkSize = 64;

    // Benchmarking control.
    //
    // For normal debugging, true is useful.
    // For timing benchmarks, false avoids measuring RMSE every outer iteration.
    bool computeRmseEachOuter = true;
    bool verbose = true;
};

struct FitResult
{
    double initialRmse = 0.0;
    double finalRmse = 0.0;
    double elapsedSeconds = 0.0;
    std::vector<double> rmseByOuterIteration;
};

class CCDPP
{
public:
    explicit CCDPP(const CCDPPConfig& config);

    FitResult fit(SparseRatings& data);

    double predict(int user, int item) const;

    double trainingRmse(const SparseRatings& data) const;

private:
    CCDPPConfig config;

    int k = 0;

    // Optimization #1:
    // Store W and H as flat contiguous arrays instead of vector<vector<double>>.
    //
    // Old:
    //   W[user][feature]
    //
    // New:
    //   W[user * k + feature]
    //
    // This improves cache locality and removes one pointer indirection.
    std::vector<double> W;
    std::vector<double> H;

    double& WAt(int user, int feature);
    const double& WAt(int user, int feature) const;

    double& HAt(int item, int feature);
    const double& HAt(int item, int feature) const;

    void initializeFactors(int numUsers, int numItems);

    void updateOneFeature(
        SparseRatings& data,
        int feature,
        std::vector<double>& u,
        std::vector<double>& v);

    void updateU(
        const SparseRatings& data,
        std::vector<double>& u,
        const std::vector<double>& v);

    void updateV(
        const SparseRatings& data,
        const std::vector<double>& u,
        std::vector<double>& v);
};