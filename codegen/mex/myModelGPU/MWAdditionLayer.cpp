/* Copyright 2017-2019 The MathWorks, Inc. */

#include "MWAdditionLayer.hpp"
#include "MWAdditionLayerImpl.hpp"

#include <stdarg.h>
#include <cassert>

MWAdditionLayer::MWAdditionLayer()
{
}

MWAdditionLayer::~MWAdditionLayer()
{    
}

void MWAdditionLayer::createAdditionLayer(MWTargetNetworkImpl* ntwk_impl, int numInputs, ...)
{
#if defined(MW_TARGET_TYPE_CUDNN) || defined(MW_TARGET_TYPE_MKLDNN) || defined(MW_TARGET_TYPE_ARMNEON)
    va_list args;
    va_start(args, numInputs);
   
    for(int i = 0; i < numInputs; i++)
    {
        MWTensor* inputTensor = va_arg(args, MWTensor*);
        setInputTensor(inputTensor, i);
    }

    int outbufIdx = va_arg(args, int);
    
    allocateOutputTensor(-1, -1, -1, -1, -1, NULL);

    getOutputTensor(0)->setopBufIndex(outbufIdx);
    m_impl = new MWAdditionLayerImpl(this, ntwk_impl);
    
#else
    va_list args;
    va_start(args, numInputs);
   
    for(int i = 0; i < numInputs; i++)
    {
        MWTensor* inputTensor = va_arg(args, MWTensor*);
        setInputTensor(inputTensor, i);
    }

    int outbufIdx = va_arg(args, int);
    
    // check that all input tensors match in size
    for(int k = 1; k < numInputs; k++)
    {
        assert(getInputTensor(0)->getHeight() == getInputTensor(k)->getHeight());
        assert(getInputTensor(0)->getWidth() == getInputTensor(k)->getWidth());
        assert(getInputTensor(0)->getChannels() == getInputTensor(k)->getChannels());
        assert(getInputTensor(0)->getBatchSize() == getInputTensor(k)->getBatchSize());
        assert(getInputTensor(0)->getSequenceLength() == getInputTensor(k)->getSequenceLength());
    }

    // allocate output                                 
    int numOutputFeatures = getInputTensor(0)->getChannels();                                      
    allocateOutputTensor(getInputTensor(0)->getHeight(), getInputTensor(0)->getWidth(), numOutputFeatures, getInputTensor(0)->getBatchSize(), getInputTensor(0)->getSequenceLength(), NULL);
                                                                                                       
    m_impl = new MWAdditionLayerImpl(this, ntwk_impl, outbufIdx);
#endif
}

void MWAdditionLayer::propagateSize()
{
#if defined(MW_TARGET_TYPE_CUDNN) || defined(MW_TARGET_TYPE_MKLDNN) || defined(MW_TARGET_TYPE_ARMNEON)
    // check that all input tensors match in size
    for(int k = 1; k < getNumInputs(); k++)
    {
        assert(getInputTensor(0)->getHeight() == getInputTensor(k)->getHeight());
        assert(getInputTensor(0)->getWidth() == getInputTensor(k)->getWidth());
        assert(getInputTensor(0)->getChannels() == getInputTensor(k)->getChannels());
        assert(getInputTensor(0)->getBatchSize() == getInputTensor(k)->getBatchSize());
        assert(getInputTensor(0)->getSequenceLength() == getInputTensor(k)->getSequenceLength());
    }
    
    resizeOutputTensor(getInputTensor()->getHeight(),
                       getInputTensor()->getWidth(),
                       getInputTensor()->getChannels(),
                       getInputTensor()->getBatchSize(),
                       getInputTensor()->getSequenceLength());

    m_impl->propagateSize();
#endif
}
