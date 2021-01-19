#include "MWElementwiseAffineLayerImpl.hpp"
#include "MWTargetNetworkImpl.hpp"
#include "cnn_api.hpp"
#include "MWKernelHeaders.hpp"
 MWElementwiseAffineLayerImpl::MWElementwiseAffineLayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl, int scale_H,  int scale_W,  int scale_C, int 
offset_H,  int offset_W,  int offset_C, bool isClipped,  int lowerbound,  int 
upperbound, const char* scale_file,  const char* offset_file, int ) : 
MWCNNLayerImpl(layer, ntwk_impl), pxmnUEWGnfCxJNuDkXAo(NULL), 
hYTzvgWajqchLzrmxjqn(NULL), rZyMIPooLjRiXLgSWDuw(scale_H), 
rwPhFWHcKnJsClVtebGW(scale_W), qquNiJHQtfSLDMNCPIBJ(scale_C), 
hqVFaqkobRNLQNgtbaai(offset_H), ikTyjLTPRBkBRlLSyxXG(offset_W), 
hpOzCTZasBMYKoXVxMDZ(offset_C), ZqQxEyCjEixByRZYMkbj(isClipped), 
crKSAZwnyiinNFYODxoN(lowerbound), vmBqKEmdajzGggqevoGl(upperbound), 
sCDdEyIOjXBVHhcakBhd(nullptr), jLmklYtHcmTxayQTpmRw(nullptr), 
qJWXFXvcpbSwehmlTNru(0), GrowsTaKrpHVUZdgZeJW(0), mJnXzwDFPTieqFtWcZIG(0) { 
loadScaleAndOffset(scale_file, offset_file); setLayerProperties(); bool 
isMatrix2d = (rZyMIPooLjRiXLgSWDuw > 1) && (rwPhFWHcKnJsClVtebGW > 1) && 
(qquNiJHQtfSLDMNCPIBJ != WawamKKnqecNqBXIyHIl); if ((!ZqQxEyCjEixByRZYMkbj) && 
(reGtUwUlPSwEenEBVIzH == hqbKXLMjsDxRQqyJEgbg ) && !isMatrix2d && 
(!eWYFXrUazhqiEIscccda->isSequenceNetwork)) { qeQuIDaHqnxGPDbPoQJF.values 
= sCDdEyIOjXBVHhcakBhd; qeQuIDaHqnxGPDbPoQJF.count = reGtUwUlPSwEenEBVIzH; 
qeQuIDaHqnxGPDbPoQJF.type = DataType::kFLOAT; 
pKmXpiCPxZwpmXlulovZ.values = nullptr; pKmXpiCPxZwpmXlulovZ.count = 
0; pKmXpiCPxZwpmXlulovZ.type = DataType::kFLOAT; 
suFVgcuEVpCOrewbJfkB.values = jLmklYtHcmTxayQTpmRw; 
suFVgcuEVpCOrewbJfkB.count = hqbKXLMjsDxRQqyJEgbg; 
suFVgcuEVpCOrewbJfkB.type = DataType::kFLOAT; ITensor* prevLayerTensor = 
getInputITensor(0); ScaleMode mode; if (reGtUwUlPSwEenEBVIzH == 1) mode = 
ScaleMode::kUNIFORM; else if (YMNbgnUYZspjMLjwcIOS == 
reGtUwUlPSwEenEBVIzH) mode = ScaleMode::kELEMENTWISE; else if (rZyMIPooLjRiXLgSWDuw 
== 1 && rwPhFWHcKnJsClVtebGW == 1 && reGtUwUlPSwEenEBVIzH == qquNiJHQtfSLDMNCPIBJ) 
mode = ScaleMode::kCHANNEL; qJWXFXvcpbSwehmlTNru = 
eWYFXrUazhqiEIscccda->network->addScale(*prevLayerTensor,  mode,  
suFVgcuEVpCOrewbJfkB, qeQuIDaHqnxGPDbPoQJF,  
pKmXpiCPxZwpmXlulovZ); assert(qJWXFXvcpbSwehmlTNru); 
qJWXFXvcpbSwehmlTNru->setName(getLayer()->getName().c_str()); 
setOpTensorPtr(qJWXFXvcpbSwehmlTNru->getOutput(0)); } else { ITensor* 
prevLayerTensor = getInputITensor(0); mJnXzwDFPTieqFtWcZIG = new 
MWPluginInterfaceImpl(this); GrowsTaKrpHVUZdgZeJW = 
eWYFXrUazhqiEIscccda->network->addPlugin(&prevLayerTensor, 1, 
*mJnXzwDFPTieqFtWcZIG); 
GrowsTaKrpHVUZdgZeJW->setName(getLayer()->getName().c_str()); 
setOpTensorPtr(GrowsTaKrpHVUZdgZeJW->getOutput(0)); } } void 
MWElementwiseAffineLayerImpl::loadScaleAndOffset(const char* 
sDWnRjToSPjYnOQzVfhS, const char* jNxFsuLXTFYGOUlfRwLW){ 
CUDA_CALL(cudaMalloc((void**)&pxmnUEWGnfCxJNuDkXAo, 
sizeof(float)*rZyMIPooLjRiXLgSWDuw*rwPhFWHcKnJsClVtebGW*qquNiJHQtfSLDMNCPIBJ)); 
CUDA_CALL(cudaMalloc((void**)&hYTzvgWajqchLzrmxjqn, 
sizeof(float)*hqVFaqkobRNLQNgtbaai*ikTyjLTPRBkBRlLSyxXG*hpOzCTZasBMYKoXVxMDZ));  
loadScale(sDWnRjToSPjYnOQzVfhS); loadOffset(jNxFsuLXTFYGOUlfRwLW); } void 
MWElementwiseAffineLayerImpl::setLayerProperties(){ WbTBQxsNsCURmwRhNTAD = 
getLayer()->getInputTensor(0)->getHeight(); XGQjNlvPuckcHnviTrkP = 
getLayer()->getInputTensor(0)->getWidth(); WawamKKnqecNqBXIyHIl = 
getLayer()->getInputTensor(0)->getChannels(); YmfPcXPXNFZDznkzKZrl = 
WbTBQxsNsCURmwRhNTAD*XGQjNlvPuckcHnviTrkP; YMNbgnUYZspjMLjwcIOS = 
YmfPcXPXNFZDznkzKZrl*WawamKKnqecNqBXIyHIl; YDoginwuwFxabuYCVqpT = 
getLayer()->getInputTensor(0)->getNumElements(); reGtUwUlPSwEenEBVIzH = 
rZyMIPooLjRiXLgSWDuw * rwPhFWHcKnJsClVtebGW * qquNiJHQtfSLDMNCPIBJ; 
hqbKXLMjsDxRQqyJEgbg = hqVFaqkobRNLQNgtbaai * ikTyjLTPRBkBRlLSyxXG * 
hpOzCTZasBMYKoXVxMDZ; assert(reGtUwUlPSwEenEBVIzH <= YDoginwuwFxabuYCVqpT); 
assert(hqbKXLMjsDxRQqyJEgbg <= YDoginwuwFxabuYCVqpT); } int 
MWElementwiseAffineLayerImpl::pluginEnqueueImpl(const void* const* inputs, 
void** outputs) { long int uTUuLVVebDakbPjXOQwp = ((YDoginwuwFxabuYCVqpT + 31) / 32) 
* 32; long int uqHugYAAqkSnCCYonqCt = (uTUuLVVebDakbPjXOQwp < 1024) ? 
uTUuLVVebDakbPjXOQwp : 1024; long int OJTEGflbxqozjWWEaUJd = 
(YDoginwuwFxabuYCVqpT + uqHugYAAqkSnCCYonqCt - 1) / 
uqHugYAAqkSnCCYonqCt; if (reGtUwUlPSwEenEBVIzH == 1) { 
scale_scalar_kernel<<<OJTEGflbxqozjWWEaUJd, uqHugYAAqkSnCCYonqCt>>>( 
(float*)inputs[0],  (float*)outputs[0], pxmnUEWGnfCxJNuDkXAo, 
YDoginwuwFxabuYCVqpT); } else if (rZyMIPooLjRiXLgSWDuw == 1 && rwPhFWHcKnJsClVtebGW 
== 1 && reGtUwUlPSwEenEBVIzH > 1) { 
scale_vector_kernel<<<OJTEGflbxqozjWWEaUJd, uqHugYAAqkSnCCYonqCt>>>( 
(float*)inputs[0],  (float*)outputs[0], pxmnUEWGnfCxJNuDkXAo, 
YmfPcXPXNFZDznkzKZrl, YMNbgnUYZspjMLjwcIOS, 
YDoginwuwFxabuYCVqpT); } else if (YMNbgnUYZspjMLjwcIOS == 
reGtUwUlPSwEenEBVIzH) {  scale_tensor3d_kernel<<<OJTEGflbxqozjWWEaUJd, 
uqHugYAAqkSnCCYonqCt>>>( (float*)inputs[0],  (float*)outputs[0], 
pxmnUEWGnfCxJNuDkXAo, XGQjNlvPuckcHnviTrkP, WbTBQxsNsCURmwRhNTAD,  
YmfPcXPXNFZDznkzKZrl,  YMNbgnUYZspjMLjwcIOS, 
YDoginwuwFxabuYCVqpT); } else { 
scale_matrix2d_kernel<<<OJTEGflbxqozjWWEaUJd, 
uqHugYAAqkSnCCYonqCt>>>( (float*)inputs[0],  (float*)outputs[0], 
pxmnUEWGnfCxJNuDkXAo, XGQjNlvPuckcHnviTrkP,  YmfPcXPXNFZDznkzKZrl,  
YMNbgnUYZspjMLjwcIOS, YDoginwuwFxabuYCVqpT); } if (hqbKXLMjsDxRQqyJEgbg 
== 1) { offset_scalar_kernel<<<OJTEGflbxqozjWWEaUJd, 
uqHugYAAqkSnCCYonqCt>>>( (float*)outputs[0],  (float*)outputs[0], 
hYTzvgWajqchLzrmxjqn, YDoginwuwFxabuYCVqpT, ZqQxEyCjEixByRZYMkbj, 
crKSAZwnyiinNFYODxoN, vmBqKEmdajzGggqevoGl); } else if (hqVFaqkobRNLQNgtbaai 
== 1 && ikTyjLTPRBkBRlLSyxXG == 1 && hqbKXLMjsDxRQqyJEgbg > 1) { 
offset_vector_kernel<<<OJTEGflbxqozjWWEaUJd, uqHugYAAqkSnCCYonqCt>>>( 
(float*)outputs[0],  (float*)outputs[0], hYTzvgWajqchLzrmxjqn, 
YmfPcXPXNFZDznkzKZrl, YMNbgnUYZspjMLjwcIOS, 
YDoginwuwFxabuYCVqpT, ZqQxEyCjEixByRZYMkbj, crKSAZwnyiinNFYODxoN, 
vmBqKEmdajzGggqevoGl); } else if (YMNbgnUYZspjMLjwcIOS == 
hqbKXLMjsDxRQqyJEgbg) { offset_tensor3d_kernel<<<OJTEGflbxqozjWWEaUJd, 
uqHugYAAqkSnCCYonqCt>>>( (float*)outputs[0],  (float*)outputs[0], 
hYTzvgWajqchLzrmxjqn, XGQjNlvPuckcHnviTrkP, WbTBQxsNsCURmwRhNTAD, 
YmfPcXPXNFZDznkzKZrl, YMNbgnUYZspjMLjwcIOS, 
YDoginwuwFxabuYCVqpT, ZqQxEyCjEixByRZYMkbj, crKSAZwnyiinNFYODxoN, 
vmBqKEmdajzGggqevoGl); } else { 
offset_matrix2d_kernel<<<OJTEGflbxqozjWWEaUJd, 
uqHugYAAqkSnCCYonqCt>>>( (float*)outputs[0],  (float*)outputs[0], 
hYTzvgWajqchLzrmxjqn, XGQjNlvPuckcHnviTrkP, YmfPcXPXNFZDznkzKZrl, 
YMNbgnUYZspjMLjwcIOS, YDoginwuwFxabuYCVqpT, ZqQxEyCjEixByRZYMkbj, 
crKSAZwnyiinNFYODxoN, vmBqKEmdajzGggqevoGl); } return 0; } void 
MWElementwiseAffineLayerImpl::loadScale(const char* sDWnRjToSPjYnOQzVfhS) { 
FILE* SZPsAnAecHGeFCSHofdG = MWCNNLayer::openBinaryFile(sDWnRjToSPjYnOQzVfhS); 
assert(SZPsAnAecHGeFCSHofdG); long int eYGiuTCCxjmoBDvVpHpn = 
rZyMIPooLjRiXLgSWDuw*rwPhFWHcKnJsClVtebGW*qquNiJHQtfSLDMNCPIBJ; sCDdEyIOjXBVHhcakBhd 
= MALLOC_CALL(sizeof(float)*eYGiuTCCxjmoBDvVpHpn); call_fread(sCDdEyIOjXBVHhcakBhd, 
sizeof(float), eYGiuTCCxjmoBDvVpHpn, SZPsAnAecHGeFCSHofdG, sDWnRjToSPjYnOQzVfhS); 
CUDA_CALL(cudaMemcpy(pxmnUEWGnfCxJNuDkXAo, sCDdEyIOjXBVHhcakBhd, 
sizeof(float)*eYGiuTCCxjmoBDvVpHpn, cudaMemcpyHostToDevice)); fclose(SZPsAnAecHGeFCSHofdG);  
} void MWElementwiseAffineLayerImpl::loadOffset(const char* 
jNxFsuLXTFYGOUlfRwLW) { FILE* SZPsAnAecHGeFCSHofdG = 
MWCNNLayer::openBinaryFile(jNxFsuLXTFYGOUlfRwLW); assert(SZPsAnAecHGeFCSHofdG); long 
int eYGiuTCCxjmoBDvVpHpn = 
hqVFaqkobRNLQNgtbaai*ikTyjLTPRBkBRlLSyxXG*hpOzCTZasBMYKoXVxMDZ; 
jLmklYtHcmTxayQTpmRw = MALLOC_CALL(sizeof(float)*eYGiuTCCxjmoBDvVpHpn); 
call_fread(jLmklYtHcmTxayQTpmRw, sizeof(float), eYGiuTCCxjmoBDvVpHpn, SZPsAnAecHGeFCSHofdG, 
jNxFsuLXTFYGOUlfRwLW); CUDA_CALL(cudaMemcpy(hYTzvgWajqchLzrmxjqn, 
jLmklYtHcmTxayQTpmRw, sizeof(float)*eYGiuTCCxjmoBDvVpHpn, cudaMemcpyHostToDevice)); 
fclose(SZPsAnAecHGeFCSHofdG);  } void MWElementwiseAffineLayerImpl::cleanup() { if 
(pxmnUEWGnfCxJNuDkXAo) { CUDA_FREE_CALL(pxmnUEWGnfCxJNuDkXAo); } if (hYTzvgWajqchLzrmxjqn) 
{ CUDA_FREE_CALL(hYTzvgWajqchLzrmxjqn); } if (sCDdEyIOjXBVHhcakBhd) 
free(sCDdEyIOjXBVHhcakBhd); if (jLmklYtHcmTxayQTpmRw) 
free(jLmklYtHcmTxayQTpmRw); }