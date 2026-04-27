#include "CCDPP.h"
#include "RatingLoader.h"
#include "SparseRatings.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

int main(int argc, char* argv[])
{
    try
    {
        std::vector<std::tuple<int, int, double>> ratings;
        int numUsers = 0;
        int numItems = 0;

        if (argc >= 2)
        {
            std::string filename = argv[1];
            int maxRatings = -1;

            if (argc >= 3)
            {
                maxRatings = std::stoi(argv[2]);
            }

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

        int rank = 2;
        double lambda = 0.1;
        int innerIterations = 5;
        int outerIterations = 10;

        CCDPP model(rank, lambda, innerIterations, outerIterations);
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
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}