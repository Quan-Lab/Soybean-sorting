/* Copyright 2017-2019 The MathWorks, Inc. */

#include "MWBatchNormalizationLayer.hpp"
#include "MWBatchNormalizationLayerImpl.hpp"
#include "MWTargetNetworkImpl.hpp"

// utils
#include <stdio.h>
#include <cassert>

MWBatchNormalizationLayer::MWBatchNormalizationLayer()
{  
}

MWBatchNormalizationLayer::~MWBatchNormalizationLayer()
{
}

void MWBatchNormalizationLayer::createBatchNormalizationLayer(MWTargetNetworkImpl* ntwk_impl,
                                                              MWTensor* MW_mangled_in,
                                                              double const MW_mangled_epsilon_in,
                                                              const char* MW_mangled_offset_file,
                                                              const char* MW_mangled_scale_file,
                                                              const char* MW_mangled_trainedMean_file,
                                                              const char* MW_mangled_trainedVariance_file,
                                                              int inPlaceOp,
                                                              int MW_mangled_numChannels,
                                                              int outbufIdx)
{
#if defined(MW_TARGET_TYPE_CUDNN) || defined(MW_TARGET_TYPE_MKLDNN) || defined(MW_TARGET_TYPE_ARMNEON)
    setInputTensor(MW_mangled_in);
    allocateOutputTensor(-1, -1, -1, -1, -1, NULL);

    getOutputTensor(0)->setopBufIndex(outbufIdx);

    numChannels = MW_mangled_numChannels;

    m_impl = new MWBatchNormalizationLayerImpl(this,
                                               ntwk_impl,
                                               MW_mangled_epsilon_in,
                                               MW_mangled_offset_file,
                                               MW_mangled_scale_file,
                                               MW_mangled_trainedMean_file,
                                               MW_mangled_trainedVariance_file,
                                               inPlaceOp,
                                               numChannels);
    
#else
    setInputTensor(MW_mangled_in);
    allocateOutputTensor(getInputTensor()->getHeight(),
                         getInputTensor()->getWidth(),
                         getInputTensor()->getChannels(),
                         getInputTensor()->getBatchSize(),
                         getInputTensor()->getSequenceLength(),
                         NULL);
    assert(getOutputTensor()->getSequenceLength() == 1);
    assert(MW_mangled_numChannels > 0);
    
    m_impl = new MWBatchNormalizationLayerImpl(this,
                                               ntwk_impl,
                                               MW_mangled_epsilon_in,
                                               MW_mangled_offset_file,
                                               MW_mangled_scale_file,
                                               MW_mangled_trainedMean_file,
                                               MW_mangled_trainedVariance_file,
                                               inPlaceOp,
                                               outbufIdx);
#endif
}

void MWBatchNormalizationLayer::propagateSize()
{
#if defined(MW_TARGET_TYPE_CUDNN) || defined(MW_TARGET_TYPE_MKLDNN) || defined(MW_TARGET_TYPE_ARMNEON)
    assert(getInputTensor()->getChannels() == numChannels);
    assert(getInputTensor()->getSequenceLength() == 1);
    
    resizeOutputTensor(getInputTensor()->getHeight(),
                       getInputTensor()->getWidth(),
                       getInputTensor()->getChannels(),
                       getInputTensor()->getBatchSize(),
                       getInputTensor()->getSequenceLength());

    m_impl->propagateSize();
#endif
}
