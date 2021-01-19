//
//  myModelGPU_initialize.h
//
//  Code generation for function 'myModelGPU_initialize'
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
CODEGEN_EXPORT_SYM void myModelGPU_initialize();

// End of code generation (myModelGPU_initialize.h)
