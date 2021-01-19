#include "MWBatchNormalizationLayer.hpp"
#include "MWBatchNormalizationLayerImpl.hpp"
#include <stdio.h>
#include <cassert> 
 __global__ __launch_bounds__(1024) void computeBNParams(float* 
pxmnUEWGnfCxJNuDkXAo,  float* hYTzvgWajqchLzrmxjqn,  float* vcOGADqMrTrPPcuYvrHS,  
float* vlzcDcTSrYXZiamsmNlx,  float RgALmBtPIZWDevjZBUHy, float* 
qLXeoFROCbISdsnwpYgl, float* sXWXkiDEKpurgeCqZLDL, float* 
niGnnRufksTFnsUUxnCj, int numVals) { long unsigned int SmibqCQPbtzycGEpwhpN = 
blockIdx.x*blockDim.x + threadIdx.x; for(; SmibqCQPbtzycGEpwhpN < numVals; 
SmibqCQPbtzycGEpwhpN+= blockDim.x*gridDim.x) { 
qLXeoFROCbISdsnwpYgl[SmibqCQPbtzycGEpwhpN] = 
pxmnUEWGnfCxJNuDkXAo[SmibqCQPbtzycGEpwhpN]/sqrt(vlzcDcTSrYXZiamsmNlx[SmibqCQPbtzycGEpwhpN] 
+ RgALmBtPIZWDevjZBUHy); sXWXkiDEKpurgeCqZLDL[SmibqCQPbtzycGEpwhpN] = 
hYTzvgWajqchLzrmxjqn[SmibqCQPbtzycGEpwhpN] - 
(vcOGADqMrTrPPcuYvrHS[SmibqCQPbtzycGEpwhpN]*qLXeoFROCbISdsnwpYgl[SmibqCQPbtzycGEpwhpN]); 
niGnnRufksTFnsUUxnCj[SmibqCQPbtzycGEpwhpN] = 1.f; } } 
MWBatchNormalizationLayerImpl::MWBatchNormalizationLayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl, double const KnbXZOZzgyMJTNGorRue, const 
char* LvFwgsCgepJxzddrrZNH, const char* LzeBIFvfQmAPApScLUyy, const char* 
MXVlAOBGDMdzlrCMvAgl, const char* MmeSJXZBVyUbgMuuFbuc, int 
, int ) : MWCNNLayerImpl(layer, ntwk_impl)  , pxmnUEWGnfCxJNuDkXAo(NULL) , 
hYTzvgWajqchLzrmxjqn(NULL) , vcOGADqMrTrPPcuYvrHS(NULL) , 
vlzcDcTSrYXZiamsmNlx(NULL) , qLXeoFROCbISdsnwpYgl(NULL) , 
sXWXkiDEKpurgeCqZLDL(NULL) , niGnnRufksTFnsUUxnCj(NULL) { 
MWBatchNormalizationLayer* BNLayer = 
static_cast<MWBatchNormalizationLayer*>(getLayer()); MWTensor* ipTensor = 
BNLayer->getInputTensor(); MWTensor* opTensor = BNLayer->getOutputTensor(); 
RgALmBtPIZWDevjZBUHy = KnbXZOZzgyMJTNGorRue; const int vVyVzWKKaCvGClCSagOb = 
2048; QJJBjzDRkBQsCLkHaADa = (ipTensor->getChannels() <= 
vVyVzWKKaCvGClCSagOb)?false:true; const size_t eczxzisTMvVejXfupdkv = 
sizeof(float)*ipTensor->getChannels(); qLXeoFROCbISdsnwpYgl = 
(float*)malloc(eczxzisTMvVejXfupdkv); sXWXkiDEKpurgeCqZLDL = 
(float*)malloc(eczxzisTMvVejXfupdkv); niGnnRufksTFnsUUxnCj = 
(float*)malloc(eczxzisTMvVejXfupdkv); if (QJJBjzDRkBQsCLkHaADa) { 
cudaMalloc(&pxmnUEWGnfCxJNuDkXAo, eczxzisTMvVejXfupdkv); 
cudaMalloc(&hYTzvgWajqchLzrmxjqn, eczxzisTMvVejXfupdkv); 
cudaMalloc(&vcOGADqMrTrPPcuYvrHS, eczxzisTMvVejXfupdkv); 
cudaMalloc(&vlzcDcTSrYXZiamsmNlx, eczxzisTMvVejXfupdkv); 
loadScale(LzeBIFvfQmAPApScLUyy); loadOffset(LvFwgsCgepJxzddrrZNH); 
loadTrainedMean(MXVlAOBGDMdzlrCMvAgl); 
loadTrainedVariance(MmeSJXZBVyUbgMuuFbuc); float* 
qMpgAuYpEDGDohMcPvRY = NULL; float* siiUXAavgXoUUOVaXdoz = NULL; float* 
nrUaguRlWYtZgNEGzlhH = NULL; cudaMalloc(&qMpgAuYpEDGDohMcPvRY, 
eczxzisTMvVejXfupdkv); cudaMalloc(&siiUXAavgXoUUOVaXdoz, 
eczxzisTMvVejXfupdkv); cudaMalloc(&nrUaguRlWYtZgNEGzlhH, 
eczxzisTMvVejXfupdkv); int fPIxBBGHjPkvmoaWByBr = ipTensor->getChannels(); int 
uTUuLVVebDakbPjXOQwp = 
std::floor(static_cast<float>(fPIxBBGHjPkvmoaWByBr)/static_cast<float>(32)) * 32; 
int uqHugYAAqkSnCCYonqCt = (uTUuLVVebDakbPjXOQwp < 1024) ? uTUuLVVebDakbPjXOQwp : 
1024; int OJTEGflbxqozjWWEaUJd = (fPIxBBGHjPkvmoaWByBr + 
uqHugYAAqkSnCCYonqCt - 1)/uqHugYAAqkSnCCYonqCt; 
computeBNParams<<<OJTEGflbxqozjWWEaUJd,uqHugYAAqkSnCCYonqCt>>>(pxmnUEWGnfCxJNuDkXAo, 
 hYTzvgWajqchLzrmxjqn,  vcOGADqMrTrPPcuYvrHS,  vlzcDcTSrYXZiamsmNlx, 
RgALmBtPIZWDevjZBUHy, qMpgAuYpEDGDohMcPvRY, siiUXAavgXoUUOVaXdoz, 
nrUaguRlWYtZgNEGzlhH, fPIxBBGHjPkvmoaWByBr); 
cudaMemcpy(qLXeoFROCbISdsnwpYgl, qMpgAuYpEDGDohMcPvRY, 
eczxzisTMvVejXfupdkv, cudaMemcpyDeviceToHost); 
cudaMemcpy(sXWXkiDEKpurgeCqZLDL, siiUXAavgXoUUOVaXdoz, 
eczxzisTMvVejXfupdkv, cudaMemcpyDeviceToHost); 
cudaMemcpy(niGnnRufksTFnsUUxnCj, nrUaguRlWYtZgNEGzlhH, 
eczxzisTMvVejXfupdkv, cudaMemcpyDeviceToHost); 
cudaFree(qMpgAuYpEDGDohMcPvRY); cudaFree(siiUXAavgXoUUOVaXdoz); 
cudaFree(nrUaguRlWYtZgNEGzlhH); } else { const size_t eczxzisTMvVejXfupdkv = 
sizeof(float)*ipTensor->getChannels(); pxmnUEWGnfCxJNuDkXAo = 
(float*)malloc(eczxzisTMvVejXfupdkv); hYTzvgWajqchLzrmxjqn = 
(float*)malloc(eczxzisTMvVejXfupdkv); vcOGADqMrTrPPcuYvrHS = 
(float*)malloc(eczxzisTMvVejXfupdkv); vlzcDcTSrYXZiamsmNlx = 
(float*)malloc(eczxzisTMvVejXfupdkv); loadScale(LzeBIFvfQmAPApScLUyy); 
loadOffset(LvFwgsCgepJxzddrrZNH); 
loadTrainedMean(MXVlAOBGDMdzlrCMvAgl); 
loadTrainedVariance(MmeSJXZBVyUbgMuuFbuc); for (int i=0; 
i<ipTensor->getChannels(); i++) { qLXeoFROCbISdsnwpYgl[i] = 
pxmnUEWGnfCxJNuDkXAo[i]/sqrt(vlzcDcTSrYXZiamsmNlx[i] + RgALmBtPIZWDevjZBUHy); 
sXWXkiDEKpurgeCqZLDL[i] = hYTzvgWajqchLzrmxjqn[i] - 
(vcOGADqMrTrPPcuYvrHS[i]*qLXeoFROCbISdsnwpYgl[i]); 
niGnnRufksTFnsUUxnCj[i] = 1.f; } } qeQuIDaHqnxGPDbPoQJF.values = 
qLXeoFROCbISdsnwpYgl; qeQuIDaHqnxGPDbPoQJF.count = 
ipTensor->getChannels(); qeQuIDaHqnxGPDbPoQJF.type = DataType::kFLOAT; 
suFVgcuEVpCOrewbJfkB.values = sXWXkiDEKpurgeCqZLDL; 
suFVgcuEVpCOrewbJfkB.count = ipTensor->getChannels(); 
suFVgcuEVpCOrewbJfkB.type = DataType::kFLOAT; 
pKmXpiCPxZwpmXlulovZ.values = niGnnRufksTFnsUUxnCj; 
pKmXpiCPxZwpmXlulovZ.count = ipTensor->getChannels(); 
pKmXpiCPxZwpmXlulovZ.type = DataType::kFLOAT; ITensor* prevLayerTensor = 
getInputITensor(0); ATYqlAsSnRELrakAbCoK = 
eWYFXrUazhqiEIscccda->network->addScale(*prevLayerTensor, ScaleMode::kCHANNEL, 
suFVgcuEVpCOrewbJfkB, qeQuIDaHqnxGPDbPoQJF, 
pKmXpiCPxZwpmXlulovZ); 
ATYqlAsSnRELrakAbCoK->setName(getLayer()->getName().c_str());  
setOpTensorPtr(ATYqlAsSnRELrakAbCoK->getOutput(0)); } 
MWBatchNormalizationLayerImpl::~MWBatchNormalizationLayerImpl() { } void 
MWBatchNormalizationLayerImpl::iLoadParamOntoGPU(char const * const 
RuGYRQXjIMQJrbgoRUxZ, int const fdiBdaeFcIDdmsgMxaJT, float* 
TYgANfbwgYWWZKKtdxCC) { FILE* SZPsAnAecHGeFCSHofdG = 
MWCNNLayer::openBinaryFile(RuGYRQXjIMQJrbgoRUxZ); assert(SZPsAnAecHGeFCSHofdG); int 
const OwscQfaoXJuSJFwXQahz = sizeof(float)*fdiBdaeFcIDdmsgMxaJT; float* 
OWgntZrUmlZXHAsNObcq = (float*)malloc(OwscQfaoXJuSJFwXQahz); 
call_fread(OWgntZrUmlZXHAsNObcq, sizeof(float), fdiBdaeFcIDdmsgMxaJT, 
SZPsAnAecHGeFCSHofdG, RuGYRQXjIMQJrbgoRUxZ); fclose(SZPsAnAecHGeFCSHofdG); 
CUDA_CALL(cudaMemcpy(TYgANfbwgYWWZKKtdxCC, OWgntZrUmlZXHAsNObcq, 
OwscQfaoXJuSJFwXQahz, cudaMemcpyHostToDevice)); free(OWgntZrUmlZXHAsNObcq); } 
void MWBatchNormalizationLayerImpl::iLoadParam(char const * const 
RuGYRQXjIMQJrbgoRUxZ, int const fdiBdaeFcIDdmsgMxaJT, float* 
EMtxAWxHxCcPIkaNDIHM) { FILE* SZPsAnAecHGeFCSHofdG = 
MWCNNLayer::openBinaryFile(RuGYRQXjIMQJrbgoRUxZ); assert(SZPsAnAecHGeFCSHofdG); 
call_fread(EMtxAWxHxCcPIkaNDIHM, sizeof(float), fdiBdaeFcIDdmsgMxaJT, 
SZPsAnAecHGeFCSHofdG, RuGYRQXjIMQJrbgoRUxZ); fclose(SZPsAnAecHGeFCSHofdG); } void 
MWBatchNormalizationLayerImpl::loadScale(const char* RuGYRQXjIMQJrbgoRUxZ) { 
MWBatchNormalizationLayer* BNLayer = 
static_cast<MWBatchNormalizationLayer*>(getLayer()); MWTensor* opTensor = 
BNLayer->getOutputTensor(); if (QJJBjzDRkBQsCLkHaADa) 
iLoadParamOntoGPU(RuGYRQXjIMQJrbgoRUxZ, opTensor->getChannels(), 
pxmnUEWGnfCxJNuDkXAo); else iLoadParam(RuGYRQXjIMQJrbgoRUxZ, 
opTensor->getChannels(), pxmnUEWGnfCxJNuDkXAo); } void 
MWBatchNormalizationLayerImpl::loadOffset(const char* RuGYRQXjIMQJrbgoRUxZ) { 
MWBatchNormalizationLayer* BNLayer = 
static_cast<MWBatchNormalizationLayer*>(getLayer()); MWTensor* opTensor = 
BNLayer->getOutputTensor(); if (QJJBjzDRkBQsCLkHaADa) 
iLoadParamOntoGPU(RuGYRQXjIMQJrbgoRUxZ, opTensor->getChannels(), 
hYTzvgWajqchLzrmxjqn); else iLoadParam(RuGYRQXjIMQJrbgoRUxZ, 
opTensor->getChannels(), hYTzvgWajqchLzrmxjqn); } void 
MWBatchNormalizationLayerImpl::loadTrainedMean(const char* RuGYRQXjIMQJrbgoRUxZ) 
{ MWBatchNormalizationLayer* BNLayer = 
static_cast<MWBatchNormalizationLayer*>(getLayer()); MWTensor* opTensor = 
BNLayer->getOutputTensor(); if (QJJBjzDRkBQsCLkHaADa) 
iLoadParamOntoGPU(RuGYRQXjIMQJrbgoRUxZ, opTensor->getChannels(), 
vcOGADqMrTrPPcuYvrHS); else iLoadParam(RuGYRQXjIMQJrbgoRUxZ, 
opTensor->getChannels(), vcOGADqMrTrPPcuYvrHS); } void 
MWBatchNormalizationLayerImpl::loadTrainedVariance(const char* 
RuGYRQXjIMQJrbgoRUxZ) { MWBatchNormalizationLayer* BNLayer = 
static_cast<MWBatchNormalizationLayer*>(getLayer()); MWTensor* opTensor = 
BNLayer->getOutputTensor(); if (QJJBjzDRkBQsCLkHaADa) 
iLoadParamOntoGPU(RuGYRQXjIMQJrbgoRUxZ, opTensor->getChannels(), 
vlzcDcTSrYXZiamsmNlx); else iLoadParam(RuGYRQXjIMQJrbgoRUxZ, 
opTensor->getChannels(), vlzcDcTSrYXZiamsmNlx); } void 
MWBatchNormalizationLayerImpl::cleanup() { if 
(QJJBjzDRkBQsCLkHaADa) { cudaFree(pxmnUEWGnfCxJNuDkXAo); 
cudaFree(hYTzvgWajqchLzrmxjqn); cudaFree(vcOGADqMrTrPPcuYvrHS); 
cudaFree(vlzcDcTSrYXZiamsmNlx); } else { free(pxmnUEWGnfCxJNuDkXAo); 
free(hYTzvgWajqchLzrmxjqn); free(vcOGADqMrTrPPcuYvrHS); 
free(vlzcDcTSrYXZiamsmNlx); } free(qLXeoFROCbISdsnwpYgl); 
free(sXWXkiDEKpurgeCqZLDL); free(niGnnRufksTFnsUUxnCj); }