#pragma once

#include "SparseRatings.h"

#include <vector>

class OptParallelCCDPP
{
public:
    OptParallelCCDPP(
        int rank,
        double lambda,
        int innerIterations,
        int outerIterations,
        float eps, int n);

    void fit(SparseRatings& data);

    double predict(int user, int item) const;

    double trainingRmse(const SparseRatings& data) const;

private:
    int k;
    double lambda;
    int innerIterations;
    int outerIterations;
    int n;
    float eps;

    std::vector<std::vector<double>> W;
    std::vector<std::vector<double>> H;

    void initializeFactors(int numUsers, int numItems);

    void updateOneFeature(SparseRatings& data, int t);

    float updateU(
        const SparseRatings& data,
        std::vector<double>& u,
        const std::vector<double>& v);

    float updateV(
        const SparseRatings& data,
        const std::vector<double>& u,
        std::vector<double>& v);
};
