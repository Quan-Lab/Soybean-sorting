/* Copyright 2017 The MathWorks, Inc. */

#ifndef __ADDITION_LAYER_HPP
#define __ADDITION_LAYER_HPP

#include "cnn_api.hpp"

/**
  *  Codegen class for AdditionLayer
  *  ElementWise Addition layer
**/
class MWAdditionLayer : public MWCNNLayer
{
  public:
    MWAdditionLayer();
    ~MWAdditionLayer();

    void createAdditionLayer(MWTargetNetworkImpl* , int numInputs, ...);
    void propagateSize();
};

#endif
