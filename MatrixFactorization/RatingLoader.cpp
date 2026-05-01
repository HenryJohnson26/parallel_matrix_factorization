#include "RatingLoader.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
    std::string trim(const std::string& s)
    {
        const std::string whitespace = " \t\r\n";

        std::size_t start = s.find_first_not_of(whitespace);
        if (start == std::string::npos)
        {
            return "";
        }

        std::size_t end = s.find_last_not_of(whitespace);
        return s.substr(start, end - start + 1);
    }

    std::vector<std::string> splitByDoubleColon(const std::string& line)
    {
        std::vector<std::string> tokens;
        std::size_t start = 0;

        while (true)
        {
            std::size_t pos = line.find("::", start);

            if (pos == std::string::npos)
            {
                tokens.push_back(trim(line.substr(start)));
                break;
            }

            tokens.push_back(trim(line.substr(start, pos - start)));
            start = pos + 2;
        }

        return tokens;
    }

    std::vector<std::string> splitByChar(const std::string& line, char delimiter)
    {
        std::vector<std::string> tokens;
        std::stringstream ss(line);
        std::string token;

        while (std::getline(ss, token, delimiter))
        {
            tokens.push_back(trim(token));
        }

        return tokens;
    }

    std::vector<std::string> splitByWhitespace(const std::string& line)
    {
        std::vector<std::string> tokens;
        std::stringstream ss(line);
        std::string token;

        while (ss >> token)
        {
            tokens.push_back(token);
        }

        return tokens;
    }

    std::vector<std::string> tokenizeLine(const std::string& line)
    {
        if (line.find("::") != std::string::npos)
        {
            return splitByDoubleColon(line);
        }

        if (line.find(',') != std::string::npos)
        {
            return splitByChar(line, ',');
        }

        if (line.find('\t') != std::string::npos)
        {
            return splitByChar(line, '\t');
        }

        return splitByWhitespace(line);
    }

    int getOrCreateIndex(std::unordered_map<int, int>& idMap, int rawId)
    {
        auto it = idMap.find(rawId);

        if (it != idMap.end())
        {
            return it->second;
        }

        int newIndex = static_cast<int>(idMap.size());
        idMap[rawId] = newIndex;
        return newIndex;
    }
}

LoadedRatings loadRatingsFromFile(
    const std::string& filename,
    bool hasHeader,
    int maxRatings)
{
    std::ifstream input(filename);

    if (!input)
    {
        throw std::runtime_error("Could not open file: " + filename);
    }

    LoadedRatings result;
    result.numUsers = 0;
    result.numItems = 0;

    std::unordered_map<int, int> userIdToIndex;
    std::unordered_map<int, int> itemIdToIndex;

    std::string line;
    int lineNumber = 0;

    while (std::getline(input, line))
    {
        ++lineNumber;

        if (line.empty())
        {
            continue;
        }

        if (hasHeader && lineNumber == 1)
        {
            continue;
        }

        if (maxRatings > 0 &&
            result.ratings.size() >= static_cast<std::size_t>(maxRatings))
        {
            break;
        }

        std::vector<std::string> tokens = tokenizeLine(line);

        if (tokens.size() < 3)
        {
            continue;
        }

        try
        {
            int rawUserId = std::stoi(tokens[0]);
            int rawItemId = std::stoi(tokens[1]);
            double rating = std::stod(tokens[2]);

            int userIndex = getOrCreateIndex(userIdToIndex, rawUserId);
            int itemIndex = getOrCreateIndex(itemIdToIndex, rawItemId);

            result.ratings.push_back({ userIndex, itemIndex, rating });
        }
        catch (const std::exception&)
        {
            // This lets us safely skip accidental headers or malformed lines.
            continue;
        }
    }

    result.numUsers = static_cast<int>(userIdToIndex.size());
    result.numItems = static_cast<int>(itemIdToIndex.size());

    return result;
}