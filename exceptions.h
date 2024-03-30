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
