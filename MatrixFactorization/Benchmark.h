#pragma once

#include "CCDPP.h"
#include "RatingLoader.h"

#include <string>
#include <vector>

struct BenchmarkRow
{
    std::string experiment;
    std::string mode;
    int threads = 1;
    int requestedRatings = 0;
    int actualRatings = 0;
    int users = 0;
    int items = 0;
    double seconds = 0.0;
    double speedup = 1.0;
    double efficiency = 1.0;
    double finalRmse = 0.0;
};

std::vector<BenchmarkRow> runStrongScalingBenchmark(
    const LoadedRatings& loaded,
    int maxRatings,
    const CCDPPConfig& baseConfig,
    int maxThreads);

std::vector<BenchmarkRow> runWeakScalingBenchmark(
    const LoadedRatings& loaded,
    int ratingsPerThread,
    const CCDPPConfig& baseConfig,
    int maxThreads);

std::vector<BenchmarkRow> runAblationBenchmark(
    const LoadedRatings& loaded,
    int maxRatings,
    const CCDPPConfig& baseConfig,
    int threads);

void writeBenchmarkCsv(
    const std::string& filename,
    const std::vector<BenchmarkRow>& rows);