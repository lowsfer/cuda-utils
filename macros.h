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
