//
//  myModelGPU_data.cu
//
//  Code generation for function 'myModelGPU_data'
//


// Include files
#include "myModelGPU_data.h"
#include "myModelGPU.h"
#include "rt_nonfinite.h"

// Variable Definitions
emlrtCTX emlrtRootTLSGlobal = NULL;
emlrtContext emlrtContextGlobal = { true,// bFirstTime
  false,                               // bInitialized
  131594U,                             // fVersionInfo
  NULL,                                // fErrorFunction
  "myModelGPU",                           // fFunctionName
  NULL,                                // fRTCallStack
  false,                               // bDebugMode
  { 3194014219U, 53452778U, 2865749887U, 3590888462U },// fSigWrd
  NULL                                 // fSigMem
};

// End of code generation (myModelGPU_data.cu)
