//
//  _coder_myModelGPU_api.cu
//
//  Code generation for function '_coder_myModelGPU_api'
//


// Include files
#include "_coder_myModelGPU_api.h"
#include "myModelGPU.h"
#include "myModelGPU_data.h"
#include "rt_nonfinite.h"

// Function Declarations
static uint8_T b_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier
  *parentId);
static uint8_T c_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier
  *msgId);
static uint8_T emlrt_marshallIn(const mxArray *x, const char_T *identifier);
static const mxArray *emlrt_marshallOut(const real_T u);

// Function Definitions
static uint8_T b_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier
  *parentId)
{
  uint8_T y;
  y = c_emlrt_marshallIn(emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}

static uint8_T c_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier
  *msgId)
{
  uint8_T ret;
  static const int32_T dims = 0;
  emlrtCheckBuiltInR2012b(emlrtRootTLSGlobal, msgId, src, "uint8", false, 0U,
    &dims);
  ret = *(uint8_T *)emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

static uint8_T emlrt_marshallIn(const mxArray *x, const char_T *identifier)
{
  uint8_T y;
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = const_cast<const char *>(identifier);
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  y = b_emlrt_marshallIn(emlrtAlias(x), &thisId);
  emlrtDestroyArray(&x);
  return y;
}

static const mxArray *emlrt_marshallOut(const real_T u)
{
  const mxArray *y;
  const mxArray *m;
  y = NULL;
  m = emlrtCreateDoubleScalar(u);
  emlrtAssign(&y, m);
  return y;
}

void myModelGPU_api(const mxArray * const prhs[1], int32_T, const mxArray *plhs[1])
{
  uint8_T x;

  // Marshall function inputs
  x = emlrt_marshallIn(emlrtAliasP(prhs[0]), "x");

  // Invoke the target function
  // Marshall function outputs
  plhs[0] = emlrt_marshallOut(myModelGPU(x));
}

// End of code generation (_coder_myModelGPU_api.cu)
