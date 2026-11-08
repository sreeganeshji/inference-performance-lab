#pragma once

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

template <typename T> std::string show(const T& value) {
    std::ostringstream out;
    out << value;
    return out.str();
}

#define CHECK(condition)                                                                                               \
    do {                                                                                                               \
        if (!(condition)) {                                                                                            \
            std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition);                         \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (false)

#define CHECK_EQ(left, right)                                                                                          \
    do {                                                                                                               \
        const auto left_value = (left);                                                                                \
        const auto right_value = (right);                                                                              \
        if (!(left_value == right_value)) {                                                                            \
            std::fprintf(stderr, "%s:%d: %s == %s failed: %s vs %s\n", __FILE__, __LINE__, #left, #right,              \
                         show(left_value).c_str(), show(right_value).c_str());                                         \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (false)
