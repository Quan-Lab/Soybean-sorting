#include <cstdlib>
#include <cassert>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include "MWConvLayerImpl.hpp"
#include "MWConvLayer.hpp"
#include "MWTargetNetworkImpl.hpp"
#include "cnn_api.hpp"
 MWConvLayerImpl::MWConvLayerImpl(MWCNNLayer* layer, MWTargetNetworkImpl* 
ntwk_impl, int filt_H, int filt_W, int numGrps, int numChnls, int numFilts, int 
HhKGcPZwrclEFnIdWerH, int HvFslbhtTwHuRNgeItfG, int GmRRxuYauzGdhIlgciAT, int 
FoMzPBFlspSYGUZHPvzd, int GVFzDZAsZZUMIMwulTWX, int GbHRuweETkejIMGyqHDI, int 
CZiiieLxAFTgpdhdjNUA, int DytNxKWQcYUHaYuuACXS, const char* 
zdBqSakTlqrPDQejGdMF, const char* OFXGTgQYmVJLJPBNAVgS, int ) : 
MWCNNLayerImpl(layer, ntwk_impl) , EBrSnWuJobWBIFNZLSZN(filt_H) , 
EFRxTzGDLCOxVeZLDhRL(filt_W) , EWUFPRDanwwTdrjmLomh(numGrps) , 
voqEJSkAwmNPuqzoiuom(NULL) , NGqpeiLeVweDRsOKEtuw(NULL) , ConvLayerT(0) { MWConvLayer* 
convLayer = static_cast<MWConvLayer*>(getLayer()); MWTensor* ipTensor = 
convLayer->getInputTensor(0); MWTensor* opTensor = 
convLayer->getOutputTensor(0); voqEJSkAwmNPuqzoiuom = 
(float*)calloc(ipTensor->getChannels() / EWUFPRDanwwTdrjmLomh * 
opTensor->getChannels() * EBrSnWuJobWBIFNZLSZN * EFRxTzGDLCOxVeZLDhRL, 
sizeof(float)); NGqpeiLeVweDRsOKEtuw = (float*)calloc(opTensor->getChannels(), 
sizeof(float)); loadWeights(zdBqSakTlqrPDQejGdMF); 
loadBias(OFXGTgQYmVJLJPBNAVgS); filt_weights.values = voqEJSkAwmNPuqzoiuom; 
filt_weights.count = ipTensor->getChannels() / EWUFPRDanwwTdrjmLomh * 
opTensor->getChannels() * EBrSnWuJobWBIFNZLSZN * EFRxTzGDLCOxVeZLDhRL; 
filt_weights.type = DataType::kFLOAT; filt_bias.values = NGqpeiLeVweDRsOKEtuw; 
filt_bias.count = opTensor->getChannels(); filt_bias.type = DataType::kFLOAT; 
ITensor* prevLayerTensor = getInputITensor(0); if((GmRRxuYauzGdhIlgciAT == 
FoMzPBFlspSYGUZHPvzd) && (GVFzDZAsZZUMIMwulTWX == GbHRuweETkejIMGyqHDI)){  
ConvLayerT = eWYFXrUazhqiEIscccda->network->addConvolution( *prevLayerTensor, 
opTensor->getChannels(), DimsHW{EBrSnWuJobWBIFNZLSZN, 
EFRxTzGDLCOxVeZLDhRL}, filt_weights, filt_bias); 
ConvLayerT->setPadding(DimsHW{GmRRxuYauzGdhIlgciAT ,GVFzDZAsZZUMIMwulTWX}); } 
else{ IPaddingLayer* iPaddingLayer = eWYFXrUazhqiEIscccda->network->addPadding( 
*prevLayerTensor, DimsHW{GmRRxuYauzGdhIlgciAT,GVFzDZAsZZUMIMwulTWX}, 
DimsHW{FoMzPBFlspSYGUZHPvzd,GbHRuweETkejIMGyqHDI}); ITensor* 
EpwuhXsRcwdqXSjBpUeO = iPaddingLayer->getOutput(0); ConvLayerT = 
eWYFXrUazhqiEIscccda->network->addConvolution( *EpwuhXsRcwdqXSjBpUeO, 
opTensor->getChannels(), DimsHW{EBrSnWuJobWBIFNZLSZN, 
EFRxTzGDLCOxVeZLDhRL}, filt_weights, filt_bias);  } 
ConvLayerT->setDilation(DimsHW{CZiiieLxAFTgpdhdjNUA, 
DytNxKWQcYUHaYuuACXS}); ConvLayerT->setStride(DimsHW{HhKGcPZwrclEFnIdWerH, 
HvFslbhtTwHuRNgeItfG}); ConvLayerT->setNbGroups(EWUFPRDanwwTdrjmLomh); 
ConvLayerT->setName(getLayer()->getName().c_str()); 
setOpTensorPtr(ConvLayerT->getOutput(0)); } void MWConvLayerImpl::cleanup() { 
free(voqEJSkAwmNPuqzoiuom); free(NGqpeiLeVweDRsOKEtuw); } void 
MWConvLayerImpl::loadWeights(const char* RtogJCavwOREhELwknZy) { MWConvLayer* 
convLayer = static_cast<MWConvLayer*>(getLayer()); FILE* SZPsAnAecHGeFCSHofdG = 
MWCNNLayer::openBinaryFile(RtogJCavwOREhELwknZy); assert(SZPsAnAecHGeFCSHofdG); int 
eYGiuTCCxjmoBDvVpHpn = convLayer->getInputTensor()->getChannels() / 
EWUFPRDanwwTdrjmLomh * convLayer->getOutputTensor()->getChannels() * 
EBrSnWuJobWBIFNZLSZN * EFRxTzGDLCOxVeZLDhRL; call_fread(voqEJSkAwmNPuqzoiuom, 
sizeof(float), eYGiuTCCxjmoBDvVpHpn, SZPsAnAecHGeFCSHofdG, RtogJCavwOREhELwknZy); 
fclose(SZPsAnAecHGeFCSHofdG); } void MWConvLayerImpl::loadBias(const char* 
RtogJCavwOREhELwknZy) { MWConvLayer* convLayer = 
static_cast<MWConvLayer*>(getLayer()); FILE* SZPsAnAecHGeFCSHofdG = 
MWCNNLayer::openBinaryFile(RtogJCavwOREhELwknZy); assert(SZPsAnAecHGeFCSHofdG); int 
eYGiuTCCxjmoBDvVpHpn = convLayer->getOutputTensor()->getChannels(); 
call_fread(NGqpeiLeVweDRsOKEtuw, sizeof(float), eYGiuTCCxjmoBDvVpHpn, SZPsAnAecHGeFCSHofdG, 
RtogJCavwOREhELwknZy); fclose(SZPsAnAecHGeFCSHofdG); }