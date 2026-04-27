#pragma once

#include "Rating.h"

#include <cstddef>
#include <tuple>
#include <vector>

class SparseRatings
{
public:
    int numUsers;
    int numItems;

    std::vector<ObservedEntry> entries;
    std::vector<std::vector<int>> entriesByUser;
    std::vector<std::vector<int>> entriesByItem;

    SparseRatings(
        int users,
        int items,
        const std::vector<std::tuple<int, int, double>>& ratings);

    std::size_t nonzeroCount() const;
};