//
//  myModelGPU_terminate.h
//
//  Code generation for function 'myModelGPU_terminate'
//


#pragma once

// Include files
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "mex.h"
#include "emlrt.h"
#include "rtwtypes.h"
#include "myModelGPU_types.h"

// Custom Header Code
#ifdef __CUDA_ARCH__
#undef printf
#endif

// Function Declarations
CODEGEN_EXPORT_SYM void myModelGPU_atexit();
CODEGEN_EXPORT_SYM void myModelGPU_terminate();

// End of code generation (myModelGPU_terminate.h)
