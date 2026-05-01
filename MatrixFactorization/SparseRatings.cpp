#include "SparseRatings.h"

#include <stdexcept>

SparseRatings::SparseRatings(
    int users,
    int items,
    const std::vector<std::tuple<int, int, double>>& ratings)
    : numUsers(users),
    numItems(items),
    entriesByUser(users),
    entriesByItem(items)
{
    entries.reserve(ratings.size());

    for (const auto& [user, item, value] : ratings)
    {
        if (user < 0 || user >= numUsers || item < 0 || item >= numItems)
        {
            throw std::runtime_error("Rating has user or item index out of range.");
        }

        int entryIndex = static_cast<int>(entries.size());

        // Since W starts at zero, the initial residual is exactly A_ij.
        entries.push_back({ user, item, value, value });

        entriesByUser[user].push_back(entryIndex);
        entriesByItem[item].push_back(entryIndex);
    }
}

std::size_t SparseRatings::nonzeroCount() const
{
    return entries.size();
}