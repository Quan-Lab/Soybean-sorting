//
//  myModelGPU_data.h
//
//  Code generation for function 'myModelGPU_data'
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

// Variable Declarations
extern emlrtCTX emlrtRootTLSGlobal;
extern emlrtContext emlrtContextGlobal;

// End of code generation (myModelGPU_data.h)
