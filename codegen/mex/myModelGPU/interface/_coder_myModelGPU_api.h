//
//  _coder_myModelGPU_api.h
//
//  Code generation for function '_coder_myModelGPU_api'
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
void myModelGPU_api(const mxArray * const prhs[1], int32_T nlhs, const mxArray
                 *plhs[1]);

// End of code generation (_coder_myModelGPU_api.h)
