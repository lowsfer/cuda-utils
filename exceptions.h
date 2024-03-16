#pragma once

#include <stdexcept>

namespace cudapp
{

class Exception : public std::runtime_error
{
public:
    Exception(const char* msg, const char* file, int line)
        : std::runtime_error{msg}
        , mFile{file}
        , mLine{line}
    {}
    const char* getFile() const {return mFile;}
    int getLine() const {return mLine;}
private:
    const char* mFile = nullptr;
    int mLine = -1;
};

class AssertionFailure : public Exception
{
public:
    using Exception::Exception;
};

} // namespace cudapp
