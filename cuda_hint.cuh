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
#include "platform.h"

#if IS_IN_IDE_PARSER
#ifdef __CUDACC__
static_assert(false, "Is in real compiler, but IDE parser is incorrectly detected.");
#endif
#define __CUDACC__ 1
#define __CUDA_ARCH__ 860
#define __CUDACC_VER_MAJOR__ 11
#define __CUDACC_VER_MINOR__ 8

unsigned __popc(unsigned);
unsigned __vabsdiffu4(unsigned, unsigned);
unsigned __dp4a(unsigned a, unsigned b, unsigned initVal);

#else
#endif

