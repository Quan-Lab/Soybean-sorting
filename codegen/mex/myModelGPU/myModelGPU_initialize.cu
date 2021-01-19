//
//  myModelGPU_initialize.cu
//
//  Code generation for function 'myModelGPU_initialize'
//


// Include files
#include "myModelGPU_initialize.h"
#include "_coder_myModelGPU_mex.h"
#include "myModelGPU.h"
#include "myModelGPU_data.h"
#include "rt_nonfinite.h"

// Function Definitions
void myModelGPU_initialize()
{
  mex_InitInfAndNan();
  mexFunctionCreateRootTLS();
  emlrtClearAllocCountR2012b(emlrtRootTLSGlobal, false, 0U, 0);
  emlrtEnterRtStackR2012b(emlrtRootTLSGlobal);
  emlrtLicenseCheckR2012b(emlrtRootTLSGlobal, "Distrib_Computing_Toolbox", 2);
  emlrtFirstTimeR2012b(emlrtRootTLSGlobal);
  cudaGetLastError();
}

// End of code generation (myModelGPU_initialize.cu)
