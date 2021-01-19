/* Copyright 2017-2018 The MathWorks, Inc. */

#ifndef __ADDITION_IMPL_HPP
#define __ADDITION_IMPL_HPP

#include "MWCNNLayerImpl.hpp"
#include "MWTargetNetworkImpl.hpp"
								
class MWAdditionLayerImpl: public MWCNNLayerImpl
{
  public:
    MWAdditionLayerImpl(MWCNNLayer* , MWTargetNetworkImpl* , int);
    ~MWAdditionLayerImpl();

private:
	IElementWiseLayer *ABtNoHVrQOgivJIJagNR;
		
};
								
#endif
