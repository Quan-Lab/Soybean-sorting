/* Copyright 2017 The MathWorks, Inc. */

#ifndef __ELEMENTWISE_AFFINE_LAYER_HPP
#define __ELEMENTWISE_AFFINE_LAYER_HPP

#include "cnn_api.hpp"

class MWElementwiseAffineLayer: public MWCNNLayer
{
  private:


  public:
    MWElementwiseAffineLayer();
    ~MWElementwiseAffineLayer();
    void createElementwiseAffineLayer(MWTargetNetworkImpl*, MWTensor*, int, int, int, int, int, int, bool, int, int, const char*, const char*, int);
    void propagateSize();
};
#endif

