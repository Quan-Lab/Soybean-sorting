//
//  myModelGPU_terminate.cu
//
//  Code generation for function 'myModelGPU_terminate'
//


// Include files
#include "myModelGPU_terminate.h"
#include "_coder_myModelGPU_mex.h"
#include "myModelGPU.h"
#include "myModelGPU_data.h"
#include "rt_nonfinite.h"

// Function Definitions
void myModelGPU_atexit()
{
  mexFunctionCreateRootTLS();
  emlrtEnterRtStackR2012b(emlrtRootTLSGlobal);
  emlrtLeaveRtStackR2012b(emlrtRootTLSGlobal);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
  emlrtExitTimeCleanup(&emlrtContextGlobal);
}

void myModelGPU_terminate()
{
  cudaError_t errCode;
  errCode = cudaGetLastError();
  if (errCode != cudaSuccess) {
    emlrtThinCUDAError(false, emlrtRootTLSGlobal);
  }

  emlrtLeaveRtStackR2012b(emlrtRootTLSGlobal);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
}

// End of code generation (myModelGPU_terminate.cu)
