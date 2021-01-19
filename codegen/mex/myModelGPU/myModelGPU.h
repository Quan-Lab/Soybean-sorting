//
//  myModelGPU.h
//
//  Code generation for function 'myModelGPU'
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
CODEGEN_EXPORT_SYM real_T myModelGPU(uint8_T x);

// End of code generation (myModelGPU.h)
