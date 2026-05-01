#pragma once

#include <string>
#include <tuple>
#include <vector>

struct LoadedRatings
{
    int numUsers;
    int numItems;
    std::vector<std::tuple<int, int, double>> ratings;
};

LoadedRatings loadRatingsFromFile(
    const std::string& filename,
    bool hasHeader = false,
    int maxRatings = -1);