#include <cstdlib>
#include <cassert>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include "MWCNNLayerImpl.hpp"
#include "MWTargetNetworkImpl.hpp"
#include "cnn_api.hpp"
#ifdef RANDOM
#include <curand.h>
 curandGenerator_t TAtTmiIniopalIZnGYzp; void 
curand_call_line_file(curandStatus_t tYWUxNVtgBrSjkBemGfF, const int 
aLQgzsOoQiSWjcCaRVKm, const char* RlwtKXlNBNTmWbDUiwqD) { if (tYWUxNVtgBrSjkBemGfF != 
CURAND_STATUS_SUCCESS) { char buffer[100]; int numElem = sprintf(buffer, 
"%d at line: %d, file: %s\n", tYWUxNVtgBrSjkBemGfF, aLQgzsOoQiSWjcCaRVKm, 
RlwtKXlNBNTmWbDUiwqD); throw std::runtime_error(buffer); } }
#endif
 void call_cuda_free(float* mem, const int aLQgzsOoQiSWjcCaRVKm, const char* 
RlwtKXlNBNTmWbDUiwqD) { if (!mem) { return; } cudaError_t tYWUxNVtgBrSjkBemGfF = 
cudaFree(mem); } float* malloc_call_line_file(size_t msize, const int 
aLQgzsOoQiSWjcCaRVKm, const char *RlwtKXlNBNTmWbDUiwqD) { float * mem = 
(float*)malloc(msize); if (!mem) { char buffer[100]; int numElem = 
sprintf(buffer, "%s at line: %d, file: %s\n", "Memory allocation failed. ", 
aLQgzsOoQiSWjcCaRVKm, RlwtKXlNBNTmWbDUiwqD); throw std::runtime_error(buffer); } return 
mem; } void cuda_call_line_file(cudaError_t tYWUxNVtgBrSjkBemGfF, const int 
aLQgzsOoQiSWjcCaRVKm, const char* RlwtKXlNBNTmWbDUiwqD) { if (tYWUxNVtgBrSjkBemGfF != 
cudaSuccess) { char buffer[100]; int numElem = sprintf(buffer, 
"Cuda Error %d(%s) at line: %d, file: %s\n", tYWUxNVtgBrSjkBemGfF, 
cudaGetErrorString(tYWUxNVtgBrSjkBemGfF), aLQgzsOoQiSWjcCaRVKm, RlwtKXlNBNTmWbDUiwqD); 
tYWUxNVtgBrSjkBemGfF = cudaGetLastError();  throw std::runtime_error(buffer); } } 
void cudnn_call_line_file(cudnnStatus_t tYWUxNVtgBrSjkBemGfF, const int 
aLQgzsOoQiSWjcCaRVKm, const char* RlwtKXlNBNTmWbDUiwqD) { if (tYWUxNVtgBrSjkBemGfF != 
CUDNN_STATUS_SUCCESS) {  char buffer[100]; int numElem = sprintf(buffer, 
"CuDNN Error %d(%s) at line: %d, file: %s\n", tYWUxNVtgBrSjkBemGfF, 
cudnnGetErrorString(tYWUxNVtgBrSjkBemGfF), aLQgzsOoQiSWjcCaRVKm, RlwtKXlNBNTmWbDUiwqD); 
throw std::runtime_error(buffer); } } 
MWCNNLayerImpl::MWCNNLayerImpl(MWCNNLayer* layer, MWTargetNetworkImpl* 
ntwk_impl) : aJTwGElOoWpBrmCfheqQ(layer) , eWYFXrUazhqiEIscccda(ntwk_impl) , 
RKrEonnJBdcnwoJXOHNM(0.0) , QlfGfPUqoazZMqFOfETJ(1.0) , QHUGvHzeHXyFElIiOliL(-1.0) , 
EiBytenrthqoQrTnOFaK(0) { } MWCNNLayerImpl::~MWCNNLayerImpl() { 
for(std::map<int, cudnnTensorDescriptor_t*>::iterator it = 
lxBZLYcHXoXUkMjfqsuo.begin(); it != lxBZLYcHXoXUkMjfqsuo.end(); ++it) { 
delete it->second; it->second = 0; } } ITensor* 
MWCNNLayerImpl::getInputITensor(int inputIdx) { MWTensor* ipTensor = 
getLayer()->getInputTensor(inputIdx); assert(ipTensor); return 
getITensor(ipTensor); } ITensor* MWCNNLayerImpl::getITensor(MWTensor* tensor) { 
if (tensor->getOwner()->getImpl() == NULL) { return 
getITensor(tensor->getOwner()->getInputTensor(0)); } else { return 
tensor->getOwner()->getImpl()->getOpTensorPtr(tensor->getSourcePortIndex()); } 
} cudnnTensorDescriptor_t* MWCNNLayerImpl::getOutputDescriptor(int index) { 
std::map<int, cudnnTensorDescriptor_t*>::iterator it = 
lxBZLYcHXoXUkMjfqsuo.find(index); if (it == lxBZLYcHXoXUkMjfqsuo.end()) { 
cudnnTensorDescriptor_t* tmp = new cudnnTensorDescriptor_t; 
lxBZLYcHXoXUkMjfqsuo[index] = tmp; assert(tmp != 0); return tmp; } else { 
assert(it->second != 0); return it->second; } } void 
MWCNNLayerImpl::deallocateOutputData(){ for (int i = 0; i < 
getLayer()->getNumOutputs(); ++i){ MWTensor* opTensor = 
getLayer()->getOutputTensor(i); float* data = opTensor->getData<float>(); if 
(data) { CUDA_FREE_CALL(data); opTensor->setData((float*)NULL); } } } 
cudnnTensorDescriptor_t* MWCNNLayerImpl::getCuDNNDescriptor(MWTensor* tensor) { 
return tensor->getOwner()->getImpl()->getOutputDescriptor( 
tensor->getSourcePortIndex()); } int MWCNNLayerImpl::pluginEnqueueImpl(const 
void* const * , void** ){ assert(false); return 0; } 
MWPluginInterfaceImpl::MWPluginInterfaceImpl(MWCNNLayerImpl* 
PfNIOWjbRyfefiYoFSmL) : m_cnnLayerImpl(PfNIOWjbRyfefiYoFSmL){} Dims 
MWPluginInterfaceImpl::getOutputDimensions(int index, const Dims* , int ) { if 
(!m_cnnLayerImpl->eWYFXrUazhqiEIscccda->isSequenceNetwork){ int 
PIyXElJqMZoWKemWyTOa = 
m_cnnLayerImpl->getLayer()->getOutputTensor(index)->getChannels(); int 
TwiaHttwApyaipMEKPSg = 
m_cnnLayerImpl->getLayer()->getOutputTensor(index)->getHeight(); int 
znJVDnWdGXAXoBVlQhwT = 
m_cnnLayerImpl->getLayer()->getOutputTensor(index)->getWidth(); return 
DimsCHW(PIyXElJqMZoWKemWyTOa, TwiaHttwApyaipMEKPSg, znJVDnWdGXAXoBVlQhwT); }
#if (NV_TENSORRT_MAJOR >= 5)
 else{ int sPCEmfHYfjaRzyVvCKeA = 
m_cnnLayerImpl->getLayer()->getOutputTensor(index)->getSequenceLength(); int 
PIyXElJqMZoWKemWyTOa = 
m_cnnLayerImpl->getLayer()->getOutputTensor(index)->getChannels(); int 
NSzdekOvRhMhRCXdWsdY = 
m_cnnLayerImpl->getLayer()->getOutputTensor(index)->getBatchSize(); return 
Dims3(sPCEmfHYfjaRzyVvCKeA, NSzdekOvRhMhRCXdWsdY, PIyXElJqMZoWKemWyTOa); }
#endif 
 } void MWPluginInterfaceImpl::configure(const Dims* inputDims, int nbInputs, 
const Dims* outputDims, int nbOutputs, int ) { assert(inputDims->nbDims == 3);  
assert(outputDims->nbDims == 3);  assert(nbInputs == 
m_cnnLayerImpl->getLayer()->getNumInputs()); assert(nbOutputs == 
m_cnnLayerImpl->getLayer()->getNumOutputs()); } int 
MWPluginInterfaceImpl::getNbOutputs() const{ return 
m_cnnLayerImpl->getLayer()->getNumOutputs(); } int 
MWPluginInterfaceImpl::enqueue(int , const void* const* inputs, void** outputs, 
void* , cudaStream_t ) { m_cnnLayerImpl->pluginEnqueueImpl(inputs,outputs); 
return 0; } MWInputLayerImpl::MWInputLayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl, int dwFpvfypaTkJiYAULzFs, int TbrveedUYuqCPPSPaVab, int 
voqEJSkAwmNPuqzoiuom, int PAwKCndEJEByqwNZnPgb, int , const char* , int ) : 
MWCNNLayerImpl(layer, ntwk_impl) { MWTensor* opTensor = 
getLayer()->getOutputTensor(0); float * PsSZzscVKwYLIATdyqkh; 
CUDA_CALL(cudaMalloc((void**)&PsSZzscVKwYLIATdyqkh, sizeof(float) * TbrveedUYuqCPPSPaVab * 
voqEJSkAwmNPuqzoiuom * PAwKCndEJEByqwNZnPgb * dwFpvfypaTkJiYAULzFs)); InputLayerITensor = 
eWYFXrUazhqiEIscccda->network->addInput( "data", DataType::kFLOAT, 
DimsCHW{PAwKCndEJEByqwNZnPgb, TbrveedUYuqCPPSPaVab, voqEJSkAwmNPuqzoiuom}); 
setOpTensorPtr(InputLayerITensor); opTensor->setData(PsSZzscVKwYLIATdyqkh); } void 
MWInputLayerImpl::cleanup() { for (int idx = 0; idx < 
aJTwGElOoWpBrmCfheqQ->getNumOutputs(); idx++) { float* data = 
aJTwGElOoWpBrmCfheqQ->getOutputTensor(idx)->getData<float>(); if (data) { 
CUDA_FREE_CALL(data); } } } MWReLULayerImpl::MWReLULayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl, int , int ) : MWCNNLayerImpl(layer, ntwk_impl) 
, iReLULayer(0) { ITensor* prevLayerTensor = getInputITensor(0); iReLULayer = 
eWYFXrUazhqiEIscccda->network->addActivation(*prevLayerTensor, 
ActivationType::kRELU); iReLULayer->setName(getLayer()->getName().c_str()); 
setOpTensorPtr(iReLULayer->getOutput(0)); } 
MWNormLayerImpl::MWNormLayerImpl(MWCNNLayer* layer, MWTargetNetworkImpl* 
ntwk_impl, unsigned IGBjAMvMJXqrubGDtvyq,  double AHTFZgpygljIqPClJcDZ,  
double AUjQjfbaYUcIYlesMFxV,  double EGziNPpAmkQdkYDEXfTU, int ) : MWCNNLayerImpl(layer, 
ntwk_impl) { ITensor* prevLayerTensor = getInputITensor(0); iNormLayer = 
eWYFXrUazhqiEIscccda->network->addLRN(*prevLayerTensor, 
IGBjAMvMJXqrubGDtvyq, AHTFZgpygljIqPClJcDZ, AUjQjfbaYUcIYlesMFxV, EGziNPpAmkQdkYDEXfTU); 
iNormLayer->setName(getLayer()->getName().c_str()); 
setOpTensorPtr(iNormLayer->getOutput(0)); } void __global__ 
__launch_bounds__(1024) MWSetDyForBackPropImpl(float * RFQXHGHdWUKqrdBFLaiy, const int 
gxwFgFgfwoXAxqyOibKF) { for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < 
gxwFgFgfwoXAxqyOibKF; i+= blockDim.x*gridDim.x) { RFQXHGHdWUKqrdBFLaiy[i] = i+1; } } 
void __global__ __launch_bounds__(1024) doMWMaxPoolingLayerImpl(float * 
UROOthsHWeMcNycRifoq, float * UIgLxHHJdliWAJIeloVl, const int 
EOuFmpbshvhRMfQlfIXQ) { for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < 
EOuFmpbshvhRMfQlfIXQ; i+= blockDim.x*gridDim.x) { if 
(static_cast<int>(UROOthsHWeMcNycRifoq[i]) != 0){ 
UIgLxHHJdliWAJIeloVl[static_cast<int>(UROOthsHWeMcNycRifoq[i])-1] = 
i; } } } MWMaxPoolingLayerImpl::MWMaxPoolingLayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl, int GtFSKuDmLreppbjSISoU,  int HKStLBswJlYYprZPPGQx,  
int HhKGcPZwrclEFnIdWerH,  int HvFslbhtTwHuRNgeItfG, int GmRRxuYauzGdhIlgciAT, int 
FoMzPBFlspSYGUZHPvzd,  int GVFzDZAsZZUMIMwulTWX, int GbHRuweETkejIMGyqHDI, bool 
INKFbkrHldYkZFmALnfC, int MW_mangled_, const std::vector<int>& ) : 
MWCNNLayerImpl(layer, ntwk_impl) , iMaxPoolingLayer(0) , 
GrowsTaKrpHVUZdgZeJW(0) , mJnXzwDFPTieqFtWcZIG(0) , 
ThkGOmtrxiMfUeOSxFsN(INKFbkrHldYkZFmALnfC) { ITensor* prevLayerTensor = 
getInputITensor(0); if (!ThkGOmtrxiMfUeOSxFsN && (GmRRxuYauzGdhIlgciAT == 
FoMzPBFlspSYGUZHPvzd) && (GVFzDZAsZZUMIMwulTWX == GbHRuweETkejIMGyqHDI)){ 
iMaxPoolingLayer = eWYFXrUazhqiEIscccda->network->addPooling( *prevLayerTensor, 
PoolingType::kMAX, DimsHW{GtFSKuDmLreppbjSISoU, HKStLBswJlYYprZPPGQx}); 
iMaxPoolingLayer->setStride(DimsHW{HhKGcPZwrclEFnIdWerH, HvFslbhtTwHuRNgeItfG}); 
iMaxPoolingLayer->setPadding(DimsHW{GmRRxuYauzGdhIlgciAT, 
GVFzDZAsZZUMIMwulTWX}); 
iMaxPoolingLayer->setName(getLayer()->getName().c_str()); 
setOpTensorPtr(iMaxPoolingLayer->getOutput(0)); } else{ 
pluginSetup(GtFSKuDmLreppbjSISoU, HKStLBswJlYYprZPPGQx, HhKGcPZwrclEFnIdWerH, 
HvFslbhtTwHuRNgeItfG, GmRRxuYauzGdhIlgciAT, GVFzDZAsZZUMIMwulTWX); 
mJnXzwDFPTieqFtWcZIG = new MWPluginInterfaceImpl(this); GrowsTaKrpHVUZdgZeJW = 
eWYFXrUazhqiEIscccda->network->addPlugin(&prevLayerTensor, 1, 
*mJnXzwDFPTieqFtWcZIG); setOpTensorPtr(GrowsTaKrpHVUZdgZeJW->getOutput(0),0); 
GrowsTaKrpHVUZdgZeJW->setName(getLayer()->getName().c_str()); if 
(ThkGOmtrxiMfUeOSxFsN) setOpTensorPtr(GrowsTaKrpHVUZdgZeJW->getOutput(1),1); 
} } float* MWMaxPoolingLayerImpl::getIndexData() { return NULL; } void 
MWMaxPoolingLayerImpl::cleanup() { if (mJnXzwDFPTieqFtWcZIG){ 
CUDNN_CALL(cudnnDestroyPoolingDescriptor(mrCdvmzPtAeVktINiAZK)); 
CUDNN_CALL(cudnnDestroyTensorDescriptor(*XOJRvKzQwSaZobhyUoOi)); 
CUDNN_CALL(cudnnDestroyTensorDescriptor(*jXNXIjpdcoiJUsfPyJJv)); } if 
(ThkGOmtrxiMfUeOSxFsN) { 
CUDNN_CALL(cudnnDestroyTensorDescriptor(*kDIJsXmMuRtKrTNwutxt)); 
CUDA_FREE_CALL(UROOthsHWeMcNycRifoq); CUDA_FREE_CALL(RFQXHGHdWUKqrdBFLaiy); } 
} void MWMaxPoolingLayerImpl::pluginSetup(int GtFSKuDmLreppbjSISoU, int 
HKStLBswJlYYprZPPGQx, int HhKGcPZwrclEFnIdWerH, int HvFslbhtTwHuRNgeItfG, int 
GmRRxuYauzGdhIlgciAT, int GVFzDZAsZZUMIMwulTWX){ MWTensor* ipTensor = 
getLayer()->getInputTensor();  
CUDNN_CALL(cudnnCreatePoolingDescriptor(&mrCdvmzPtAeVktINiAZK)); 
CUDNN_CALL(cudnnSetPooling2dDescriptor(mrCdvmzPtAeVktINiAZK, CUDNN_POOLING_MAX, 
CUDNN_NOT_PROPAGATE_NAN, GtFSKuDmLreppbjSISoU, HKStLBswJlYYprZPPGQx, 
GmRRxuYauzGdhIlgciAT, GVFzDZAsZZUMIMwulTWX, HhKGcPZwrclEFnIdWerH, 
HvFslbhtTwHuRNgeItfG)); XOJRvKzQwSaZobhyUoOi = new cudnnTensorDescriptor_t; 
CUDNN_CALL(cudnnCreateTensorDescriptor(XOJRvKzQwSaZobhyUoOi)); 
CUDNN_CALL(cudnnSetTensor4dDescriptor(*XOJRvKzQwSaZobhyUoOi, CUDNN_TENSOR_NCHW,  
CUDNN_DATA_FLOAT, ipTensor->getBatchSize(),  ipTensor->getChannels(),  
ipTensor->getHeight(),  ipTensor->getWidth()));  int dwFpvfypaTkJiYAULzFs, 
PAwKCndEJEByqwNZnPgb, TbrveedUYuqCPPSPaVab, voqEJSkAwmNPuqzoiuom; 
CUDNN_CALL(cudnnGetPooling2dForwardOutputDim(mrCdvmzPtAeVktINiAZK, 
*XOJRvKzQwSaZobhyUoOi, &dwFpvfypaTkJiYAULzFs ,&PAwKCndEJEByqwNZnPgb, &TbrveedUYuqCPPSPaVab, 
&voqEJSkAwmNPuqzoiuom)); TbrveedUYuqCPPSPaVab = getLayer()->getOutputTensor(0)->getHeight(); 
voqEJSkAwmNPuqzoiuom = getLayer()->getOutputTensor(0)->getWidth(); jXNXIjpdcoiJUsfPyJJv = 
new cudnnTensorDescriptor_t; 
CUDNN_CALL(cudnnCreateTensorDescriptor(jXNXIjpdcoiJUsfPyJJv)); 
CUDNN_CALL(cudnnSetTensor4dDescriptor(*jXNXIjpdcoiJUsfPyJJv, CUDNN_TENSOR_NCHW, 
CUDNN_DATA_FLOAT, dwFpvfypaTkJiYAULzFs, PAwKCndEJEByqwNZnPgb, TbrveedUYuqCPPSPaVab, voqEJSkAwmNPuqzoiuom)); if 
(ThkGOmtrxiMfUeOSxFsN){ kDIJsXmMuRtKrTNwutxt = new cudnnTensorDescriptor_t; 
CUDNN_CALL(cudnnCreateTensorDescriptor(kDIJsXmMuRtKrTNwutxt)); 
CUDNN_CALL(cudnnSetTensor4dDescriptor(*kDIJsXmMuRtKrTNwutxt, CUDNN_TENSOR_NCHW, 
CUDNN_DATA_FLOAT, dwFpvfypaTkJiYAULzFs, PAwKCndEJEByqwNZnPgb, TbrveedUYuqCPPSPaVab, voqEJSkAwmNPuqzoiuom)); 
assert((PAwKCndEJEByqwNZnPgb == ipTensor->getChannels()) && (dwFpvfypaTkJiYAULzFs == 
ipTensor->getBatchSize()));  fPIxBBGHjPkvmoaWByBr = 
(ipTensor->getHeight())*(ipTensor->getWidth())*(ipTensor->getChannels())*(ipTensor->getBatchSize()); 
CUDA_CALL(cudaMalloc((void**)&UROOthsHWeMcNycRifoq, 
sizeof(float)*fPIxBBGHjPkvmoaWByBr)); gxwFgFgfwoXAxqyOibKF = 
voqEJSkAwmNPuqzoiuom*TbrveedUYuqCPPSPaVab*PAwKCndEJEByqwNZnPgb*dwFpvfypaTkJiYAULzFs; 
CUDA_CALL(cudaMalloc((void**)&RFQXHGHdWUKqrdBFLaiy, 
sizeof(float)*gxwFgFgfwoXAxqyOibKF)); int uqHugYAAqkSnCCYonqCt = 
(gxwFgFgfwoXAxqyOibKF < 1024) ? gxwFgFgfwoXAxqyOibKF : 1024; int 
OJTEGflbxqozjWWEaUJd = (gxwFgFgfwoXAxqyOibKF + uqHugYAAqkSnCCYonqCt - 
1)/uqHugYAAqkSnCCYonqCt; 
MWSetDyForBackPropImpl<<<OJTEGflbxqozjWWEaUJd, 
uqHugYAAqkSnCCYonqCt>>>( RFQXHGHdWUKqrdBFLaiy, gxwFgFgfwoXAxqyOibKF); } } int 
MWMaxPoolingLayerImpl::pluginEnqueueImpl(const void* const * inputs, void** 
outputs){ 
CUDNN_CALL(cudnnPoolingForward(*eWYFXrUazhqiEIscccda->getCudnnHandle(), 
mrCdvmzPtAeVktINiAZK, getOnePtr(), *XOJRvKzQwSaZobhyUoOi, (float*)inputs[0], 
getZeroPtr(), *jXNXIjpdcoiJUsfPyJJv, (float*)outputs[0])); if 
(ThkGOmtrxiMfUeOSxFsN) { MWTensor* ipTensor = getLayer()->getInputTensor(); 
CUDNN_CALL(cudnnPoolingBackward(*eWYFXrUazhqiEIscccda->getCudnnHandle(), 
mrCdvmzPtAeVktINiAZK, getOnePtr(), *jXNXIjpdcoiJUsfPyJJv, (float*)outputs[0], 
*jXNXIjpdcoiJUsfPyJJv, RFQXHGHdWUKqrdBFLaiy, *XOJRvKzQwSaZobhyUoOi, (float*)inputs[0], 
getZeroPtr(), *XOJRvKzQwSaZobhyUoOi, UROOthsHWeMcNycRifoq)); int 
uqHugYAAqkSnCCYonqCt = (fPIxBBGHjPkvmoaWByBr < 1024) ? fPIxBBGHjPkvmoaWByBr : 
1024; int OJTEGflbxqozjWWEaUJd = (fPIxBBGHjPkvmoaWByBr + 
uqHugYAAqkSnCCYonqCt - 1)/uqHugYAAqkSnCCYonqCt; 
doMWMaxPoolingLayerImpl<<<OJTEGflbxqozjWWEaUJd, 
uqHugYAAqkSnCCYonqCt>>>( UROOthsHWeMcNycRifoq, 
(float*)outputs[1], fPIxBBGHjPkvmoaWByBr); } return 0; } 
MWFCLayerImpl::MWFCLayerImpl(MWCNNLayer* layer, MWTargetNetworkImpl* ntwk_impl, 
int EfvWctmlsWAPsxXgdKWf, const char* zdBqSakTlqrPDQejGdMF,  const 
char* OFXGTgQYmVJLJPBNAVgS, int ) : MWCNNLayerImpl(layer, ntwk_impl) , 
iFCLayer(0) { MWTensor* opTensor = getLayer()->getOutputTensor(0); MWTensor* 
ipTensor = getLayer()->getInputTensor(0); voqEJSkAwmNPuqzoiuom = 
(float*)calloc(EfvWctmlsWAPsxXgdKWf * opTensor->getChannels(), 
sizeof(float)); NGqpeiLeVweDRsOKEtuw = (float*)calloc(opTensor->getChannels(), 
sizeof(float)); int eYGiuTCCxjmoBDvVpHpn = EfvWctmlsWAPsxXgdKWf * 
opTensor->getChannels();  loadWeights(eYGiuTCCxjmoBDvVpHpn, zdBqSakTlqrPDQejGdMF); 
loadBias(OFXGTgQYmVJLJPBNAVgS); ITensor* prevLayerITensor = getInputITensor(0); 
filt_weights.values = voqEJSkAwmNPuqzoiuom; filt_weights.count = 
EfvWctmlsWAPsxXgdKWf * opTensor->getChannels(); filt_weights.type = 
DataType::kFLOAT; filt_bias.values = NGqpeiLeVweDRsOKEtuw; filt_bias.count = 
opTensor->getChannels(); filt_bias.type = DataType::kFLOAT; if 
(!eWYFXrUazhqiEIscccda->isSequenceNetwork){ iFCLayer = 
eWYFXrUazhqiEIscccda->network->addFullyConnected( *prevLayerITensor, 
opTensor->getChannels(), filt_weights, filt_bias); 
iFCLayer->setName(getLayer()->getName().c_str()); 
setOpTensorPtr(iFCLayer->getOutput(0)); }
#if (NV_TENSORRT_MAJOR >= 5)
 else{ auto shuffleLayer = 
eWYFXrUazhqiEIscccda->network->addShuffle(*prevLayerITensor); 
assert(shuffleLayer); shuffleLayer->setFirstTranspose(Permutation{1, 0, 2}); 
auto fcwts = eWYFXrUazhqiEIscccda->network->addConstant(Dims3(1, 
opTensor->getChannels(), EfvWctmlsWAPsxXgdKWf), filt_weights);
#if (NV_TENSORRT_MAJOR >= 5 && NV_TENSORRT_MINOR >= 1)
 auto matrixMultLayer = eWYFXrUazhqiEIscccda->network->addMatrixMultiply( 
*fcwts->getOutput(0), MatrixOperation::kNONE, *shuffleLayer->getOutput(0), MatrixOperation::kTRANSPOSE);
#else
 auto matrixMultLayer = eWYFXrUazhqiEIscccda->network->addMatrixMultiply( 
*fcwts->getOutput(0), false, *shuffleLayer->getOutput(0), true);
#endif
 assert(matrixMultLayer != nullptr); auto fcbias = 
eWYFXrUazhqiEIscccda->network->addConstant(Dims3(1, opTensor->getChannels(), 1), 
filt_bias); auto elementWiseLayer = 
eWYFXrUazhqiEIscccda->network->addElementWise(*matrixMultLayer->getOutput(0), 
*fcbias->getOutput(0), ElementWiseOperation::kSUM); assert(elementWiseLayer != 
nullptr); shuffleLayer = 
eWYFXrUazhqiEIscccda->network->addShuffle(*elementWiseLayer->getOutput(0)); 
assert(shuffleLayer); shuffleLayer->setFirstTranspose(Permutation{2, 0, 1}); 
setOpTensorPtr(shuffleLayer->getOutput(0)); }
#endif
 } void MWFCLayerImpl::loadWeights(int eYGiuTCCxjmoBDvVpHpn, const char* 
RtogJCavwOREhELwknZy) { MWFCLayer* fcLayer = 
static_cast<MWFCLayer*>(getLayer()); MWTensor* ipTensor = 
fcLayer->getInputTensor(0); FILE* SZPsAnAecHGeFCSHofdG = 
MWCNNLayer::openBinaryFile(RtogJCavwOREhELwknZy); assert(SZPsAnAecHGeFCSHofdG); 
call_fread(voqEJSkAwmNPuqzoiuom, sizeof(float), eYGiuTCCxjmoBDvVpHpn, SZPsAnAecHGeFCSHofdG, 
RtogJCavwOREhELwknZy); if (ipTensor->getHeight() != 1 && ipTensor->getWidth() != 
1) { float* OuTwywxKeMgznElXdjGp = (float*)malloc(sizeof(float) * 
ipTensor->getHeight() * ipTensor->getWidth()); for (int k = 0; k < 
eYGiuTCCxjmoBDvVpHpn / ipTensor->getHeight() / ipTensor->getWidth(); k++) { for (int 
i = 0; i < ipTensor->getHeight() * ipTensor->getWidth(); i++) { 
OuTwywxKeMgznElXdjGp[i] = voqEJSkAwmNPuqzoiuom[k * ipTensor->getHeight() * 
ipTensor->getWidth() + i]; } for (int j = 0; j < ipTensor->getHeight(); j++) 
for (int i = 0; i < ipTensor->getWidth(); i++) { voqEJSkAwmNPuqzoiuom[k * 
ipTensor->getHeight() * ipTensor->getWidth() + j * ipTensor->getWidth() + i] = 
OuTwywxKeMgznElXdjGp[j + i * ipTensor->getHeight()]; } } 
free(OuTwywxKeMgznElXdjGp); } fclose(SZPsAnAecHGeFCSHofdG); } void 
MWFCLayerImpl::loadBias(const char* RtogJCavwOREhELwknZy) { MWTensor* opTensor = 
getLayer()->getOutputTensor(0); FILE* SZPsAnAecHGeFCSHofdG = 
MWCNNLayer::openBinaryFile(RtogJCavwOREhELwknZy); assert(SZPsAnAecHGeFCSHofdG); int 
eYGiuTCCxjmoBDvVpHpn = opTensor->getChannels();  call_fread(NGqpeiLeVweDRsOKEtuw, 
sizeof(float), eYGiuTCCxjmoBDvVpHpn, SZPsAnAecHGeFCSHofdG, RtogJCavwOREhELwknZy); 
fclose(SZPsAnAecHGeFCSHofdG); } void MWFCLayerImpl::cleanup() { free(voqEJSkAwmNPuqzoiuom); 
free(NGqpeiLeVweDRsOKEtuw); } MWSoftmaxLayerImpl::MWSoftmaxLayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl, int ) : MWCNNLayerImpl(layer, ntwk_impl) , 
iSoftmaxLayer(0) { MWTensor* opTensor = getLayer()->getOutputTensor(0); 
ITensor* prevLayerTensor = getInputITensor(0); if 
(!eWYFXrUazhqiEIscccda->isSequenceNetwork){ iSoftmaxLayer = 
eWYFXrUazhqiEIscccda->network->addSoftMax(*prevLayerTensor); }
#if (NV_TENSORRT_MAJOR >= 5) 
 else{ iSoftmaxLayer = 
eWYFXrUazhqiEIscccda->network->addSoftMax(*prevLayerTensor); 
iSoftmaxLayer->setAxes(1<<2); }
#endif
 iSoftmaxLayer->setName(getLayer()->getName().c_str()); 
setOpTensorPtr(iSoftmaxLayer->getOutput(0)); } 
MWOutputLayerImpl::MWOutputLayerImpl(MWCNNLayer* layer, MWTargetNetworkImpl* 
ntwk_impl, int ) : MWCNNLayerImpl(layer, ntwk_impl) { MWTensor* opTensor = 
getLayer()->getOutputTensor(0); float * PsSZzscVKwYLIATdyqkh; 
CUDA_CALL(cudaMalloc((void**)&PsSZzscVKwYLIATdyqkh, sizeof(float) * 
opTensor->getNumElements())); ITensor* prevLayerTensor = getInputITensor(0); 
setOpTensorPtr(prevLayerTensor); opTensor->setData(PsSZzscVKwYLIATdyqkh); } void 
MWOutputLayerImpl::cleanup() { for (int idx = 0; idx < 
aJTwGElOoWpBrmCfheqQ->getNumOutputs(); idx++) { float* data = 
aJTwGElOoWpBrmCfheqQ->getOutputTensor(idx)->getData<float>(); if (data) { 
CUDA_FREE_CALL(data); } } } 
MWAvgPoolingLayerImpl::MWAvgPoolingLayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl, int GtFSKuDmLreppbjSISoU,  int HKStLBswJlYYprZPPGQx,  
int HhKGcPZwrclEFnIdWerH,  int HvFslbhtTwHuRNgeItfG,  int GmRRxuYauzGdhIlgciAT,  
int FoMzPBFlspSYGUZHPvzd,  int GVFzDZAsZZUMIMwulTWX, int GbHRuweETkejIMGyqHDI, 
int ) : MWCNNLayerImpl(layer, ntwk_impl) , iAvgPoolingLayer(0) { ITensor* 
prevLayerTensor = getInputITensor(0); if((GmRRxuYauzGdhIlgciAT == 
FoMzPBFlspSYGUZHPvzd) && (GVFzDZAsZZUMIMwulTWX == GbHRuweETkejIMGyqHDI)){  
iAvgPoolingLayer = eWYFXrUazhqiEIscccda->network->addPooling( *prevLayerTensor, 
PoolingType::kAVERAGE, DimsHW{GtFSKuDmLreppbjSISoU, HKStLBswJlYYprZPPGQx}); 
iAvgPoolingLayer->setPadding(DimsHW{GmRRxuYauzGdhIlgciAT, 
GVFzDZAsZZUMIMwulTWX}); } else { IPaddingLayer* iPaddingLayer = 
eWYFXrUazhqiEIscccda->network->addPadding( *prevLayerTensor, 
DimsHW{GmRRxuYauzGdhIlgciAT,GVFzDZAsZZUMIMwulTWX}, 
DimsHW{FoMzPBFlspSYGUZHPvzd,GbHRuweETkejIMGyqHDI}); ITensor* 
EpwuhXsRcwdqXSjBpUeO = iPaddingLayer->getOutput(0); iAvgPoolingLayer = 
eWYFXrUazhqiEIscccda->network->addPooling( *EpwuhXsRcwdqXSjBpUeO, 
PoolingType::kAVERAGE, DimsHW{GtFSKuDmLreppbjSISoU, HKStLBswJlYYprZPPGQx});  } 
iAvgPoolingLayer->setStride(DimsHW{HhKGcPZwrclEFnIdWerH, HvFslbhtTwHuRNgeItfG}); 
iAvgPoolingLayer->setAverageCountExcludesPadding(false); 
iAvgPoolingLayer->setName(getLayer()->getName().c_str()); 
setOpTensorPtr(iAvgPoolingLayer->getOutput(0)); }