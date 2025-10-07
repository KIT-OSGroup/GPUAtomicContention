#pragma once

#include <chrono>
#include <concepts>
#include <exception>
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <ostream>
#include <string>
#include <string_view>
#include <thread>
#include <unistd.h>

#include "utility.hpp"

#define ASSERT(cond) test::test_assert((cond), "ASSERT(" #cond ")", __FILE__, __LINE__)
#define FAIL(message) test::test_assert(false, "FAIL(" #message ")", __FILE__, __LINE__)

#define COLOR_WHITE "\e[0;37m"
#define COLOR_RED "\e[0;31m"
#define COLOR_GREEN "\e[0;32m"

namespace test {
    enum class Completion {
        PASS,
        FAILURE
    };

    class test_exception : public std::exception {
    public:
        test_exception(std::string &&message)
            : m_message(std::move(message))
        {}

        virtual const char *what() const throw() override {
            return m_message.c_str();
        }

    private:
        std::string m_message{};
    };

    template <std::regular_invocable Functor>
    Completion test(std::string_view name, const Functor &test) {
        try {
            test();
            std::cout << COLOR_GREEN << "[PASS] " << name << "" << COLOR_WHITE << std::endl;
            return Completion::PASS;
        } catch (test_exception exception) {
            std::cout << COLOR_RED << "[FAIL] " << name << std::endl;
            std::cout << "    " << exception.what() << COLOR_WHITE << std::endl;
            return Completion::FAILURE;
        }
    }

    static void test_assert(bool condition, std::string message, std::string file, int line) {
        if (!condition) {
            throw test_exception(file + ":" + std::to_string(line) + ": " + message);
        }
    }
}
