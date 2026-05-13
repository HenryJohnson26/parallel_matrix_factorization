
#include "CCDPP.h"
#include "ParallelCCDPP.h"
#include "OptParallelCCDPP.h"
#include "RatingLoader.h"
#include "SparseRatings.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
    void printUsage()
    {
        std::cout
            << "Usage examples:\n"
            << "  ./ccdpp [ratings.dat] [maxRatings] [-s|-p|-o] [-n threads] [-e epsilon]\n"
            << "          [-r rank] [-t innerIterations] [-i outerIterations] [--no-predictions]\n\n"
            << "Examples:\n"
            << "  ./ccdpp ratings.dat 1000000 -s -r 2 -t 5 -i 10 --no-predictions\n"
            << "  ./ccdpp ratings.dat 1000000 -p -n 4 -r 2 -t 5 -i 10 --no-predictions\n"
            << "  ./ccdpp ratings.dat 1000000 -o -n 4 -e 0.001 -r 2 -t 5 -i 10 --no-predictions\n";
    }
}

int main(int argc, char* argv[])
{
    try
    {
        bool useParallel = false;
        bool useOpt = false;
        bool printPredictionsFlag = true;

        std::string filename;
        int maxRatings = -1;
        int numThreads = 4;
        float eps = 1e-3f;

        int rank = 2;
        double lambda = 0.1;
        int innerIterations = 5;
        int outerIterations = 10;

        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];

            if (arg == "-h" || arg == "--help")
            {
                printUsage();
                return 0;
            }
            else if (arg == "-p")
            {
                useParallel = true;
            }
            else if (arg == "-s")
            {
                useParallel = false;
                useOpt = false;
            }
            else if (arg == "-o")
            {
                useOpt = true;
                useParallel = true;
            }
            else if (arg == "-n" && i + 1 < argc)
            {
                numThreads = std::stoi(argv[++i]);
            }
            else if (arg == "-e" && i + 1 < argc)
            {
                eps = std::stof(argv[++i]);
            }
            else if (arg == "-r" && i + 1 < argc)
            {
                rank = std::stoi(argv[++i]);
            }
            else if (arg == "-t" && i + 1 < argc)
            {
                innerIterations = std::stoi(argv[++i]);
            }
            else if (arg == "-i" && i + 1 < argc)
            {
                outerIterations = std::stoi(argv[++i]);
            }
            else if (arg == "--no-predictions")
            {
                printPredictionsFlag = false;
            }
            else if (filename.empty())
            {
                filename = arg;
            }
            else
            {
                maxRatings = std::stoi(arg);
            }
        }

#ifdef _OPENMP
        if (useParallel)
        {
            omp_set_num_threads(numThreads);
        }
#endif

        std::vector<std::tuple<int, int, double>> ratings;
        int numUsers = 0;
        int numItems = 0;

        if (!filename.empty())
        {
            LoadedRatings loaded = loadRatingsFromFile(filename, false, maxRatings);
            ratings = loaded.ratings;
            numUsers = loaded.numUsers;
            numItems = loaded.numItems;
            std::cout << "Loaded ratings from file: " << filename << "\n";
        }
        else
        {
            std::cout << "No input file provided. Using small hardcoded test matrix.\n";
            ratings = {
                {0, 0, 5.0}, {0, 1, 3.0}, {0, 3, 1.0},
                {1, 0, 4.0}, {1, 3, 1.0},
                {2, 0, 1.0}, {2, 1, 1.0}, {2, 3, 5.0}, {2, 4, 4.0},
                {3, 1, 1.0}, {3, 4, 5.0}
            };
            numUsers = 4;
            numItems = 5;
        }

        SparseRatings data(numUsers, numItems, ratings);
        int count = std::min<int>(20, static_cast<int>(data.entries.size()));

        auto printPredictions = [&](auto& model)
        {
            if (!printPredictionsFlag)
            {
                return;
            }

            std::cout << "\nFirst few predictions for observed ratings:\n";
            for (int i = 0; i < count; ++i)
            {
                const ObservedEntry& entry = data.entries[i];
                std::cout << "user " << entry.user
                    << ", item " << entry.item
                    << ", actual " << entry.rating
                    << ", predicted " << model.predict(entry.user, entry.item)
                    << "\n";
            }
        };

        std::cout << "Experiment config:\n"
            << "  rank=" << rank << "\n"
            << "  lambda=" << lambda << "\n"
            << "  innerIterations(T)=" << innerIterations << "\n"
            << "  outerIterations=" << outerIterations << "\n"
            << "  threads=" << numThreads << "\n"
            << "  epsilon=" << eps << "\n";

        auto start = std::chrono::steady_clock::now();

        if (useOpt)
        {
            std::cout << "Using optimized parallel implementation (threads=" << numThreads << ", eps=" << eps << ")\n";
            OptParallelCCDPP model(rank, lambda, innerIterations, outerIterations, eps, numThreads);
            model.fit(data);
            printPredictions(model);
        }
        else if (useParallel)
        {
            std::cout << "Using parallel implementation (threads=" << numThreads << ")\n";
            ParallelCCDPP model(rank, lambda, innerIterations, outerIterations);
            model.fit(data);
            printPredictions(model);
        }
        else
        {
            std::cout << "Using serial implementation\n";
            CCDPP model(rank, lambda, innerIterations, outerIterations);
            model.fit(data);
            printPredictions(model);
        }

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Total elapsed seconds: " << elapsed.count() << "\n";
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
