//
//  _coder_myModelGPU_mex.cu
//
//  Code generation for function '_coder_myModelGPU_mex'
//


// Include files
#include "_coder_myModelGPU_mex.h"
#include "_coder_myModelGPU_api.h"
#include "myModelGPU.h"
#include "myModelGPU_data.h"
#include "myModelGPU_initialize.h"
#include "myModelGPU_terminate.h"

// Function Declarations
MEXFUNCTION_LINKAGE void myModelGPU_mexFunction(int32_T nlhs, mxArray *plhs[1],
  int32_T nrhs, const mxArray *prhs[1]);

// Function Definitions
void myModelGPU_mexFunction(int32_T nlhs, mxArray *plhs[1], int32_T nrhs, const
  mxArray *prhs[1])
{
  const mxArray *outputs[1];

  // Check for proper number of arguments.
  if (nrhs != 1) {
    emlrtErrMsgIdAndTxt(emlrtRootTLSGlobal, "EMLRT:runTime:WrongNumberOfInputs",
                        5, 12, 1, 4, 7, "myModelGPU");
  }

  if (nlhs > 1) {
    emlrtErrMsgIdAndTxt(emlrtRootTLSGlobal,
                        "EMLRT:runTime:TooManyOutputArguments", 3, 4, 7,
                        "myModelGPU");
  }

  // Call the function.
  myModelGPU_api(prhs, nlhs, outputs);

  // Copy over outputs to the caller.
  emlrtReturnArrays(1, plhs, outputs);
}

void mexFunction(int32_T nlhs, mxArray *plhs[], int32_T nrhs, const mxArray
                 *prhs[])
{
  mexAtExit(&myModelGPU_atexit);

  // Module initialization.
  myModelGPU_initialize();

  // Dispatch the entry-point.
  myModelGPU_mexFunction(nlhs, plhs, nrhs, prhs);

  // Module termination.
  myModelGPU_terminate();
}

emlrtCTX mexFunctionCreateRootTLS()
{
  emlrtCreateRootTLS(&emlrtRootTLSGlobal, &emlrtContextGlobal, NULL, 1);
  return emlrtRootTLSGlobal;
}

// End of code generation (_coder_myModelGPU_mex.cu)
