/*
Copyright [2024] [Yao Yao]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*/

#pragma once

#include "exceptions.h"

namespace cudapp
{
namespace impl
{
[[noreturn]] inline void throwException(const char* msg, const char* file, int line){
    throw Exception{msg, file, line};
}
inline void assertImpl(bool condition, const char* expr, const char* file, int line){
    if (!condition){
        printf("Error %s:%d: failed assertion %s\n", file, line, expr);
        throw AssertionFailure{expr, file, line};
    }
}
inline void hopeImpl(bool condition, const char* expr, const char* file, int line){
    if (!condition){
        printf("Warning: %s at %s:%d\n", expr, file, line);
    }
}

} // namespace impl
} // namespace cudapp

#define DIE(msg) cudapp::impl::throwException(msg, __FILE__, __LINE__)
#define ASSERT(expr) cudapp::impl::assertImpl((expr), #expr, __FILE__, __LINE__)

#define HOPE(expr) cudapp::impl::hopeImpl((expr), #expr, __FILE__, __LINE__)
#define NOEXCEPT(expr) \
    do{\
        try{\
            expr;\
        }\
        catch(...){\
            std::terminate();\
        }\
    }while(false)
