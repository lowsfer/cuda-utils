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

