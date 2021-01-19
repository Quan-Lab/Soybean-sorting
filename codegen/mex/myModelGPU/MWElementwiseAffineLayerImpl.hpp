/* Copyright 2019 The MathWorks, Inc. */

#ifndef __ELEMENTWISE_AFFINE_LAYER_IMPL_HPP
#define __ELEMENTWISE_AFFINE_LAYER_IMPL_HPP

#include "MWCNNLayerImpl.hpp"

/**
 *  Codegen class for Elementwise affine layer
 **/
class MWElementwiseAffineLayerImpl : public MWCNNLayerImpl {
  public:
    MWElementwiseAffineLayerImpl(MWCNNLayer* layer,
                                 MWTargetNetworkImpl* ntwk_impl,
                                 int scale_H,
                                 int scale_W,
                                 int scale_C,
                                 int offset_H,
                                 int offset_W,
                                 int offset_C,
                                 bool isClipped,
                                 int lowerbound,
                                 int upperbound,
                                 const char* scale_file,
                                 const char* offsetfile,
                                 int outbufIdx);
    
    ~MWElementwiseAffineLayerImpl() {
    }

    int pluginEnqueueImpl(const void* const*, void**);

    void cleanup();

  private:
    void loadScale(const char*);
    void loadOffset(const char*);
    float* pxmnUEWGnfCxJNuDkXAo;
    float* hYTzvgWajqchLzrmxjqn;
    double rZyMIPooLjRiXLgSWDuw;
    double rwPhFWHcKnJsClVtebGW;
    double qquNiJHQtfSLDMNCPIBJ;
    double hqVFaqkobRNLQNgtbaai;
    double ikTyjLTPRBkBRlLSyxXG;
    double hpOzCTZasBMYKoXVxMDZ;
    bool ZqQxEyCjEixByRZYMkbj;
    int crKSAZwnyiinNFYODxoN;
    int vmBqKEmdajzGggqevoGl;


    IPluginLayer* GrowsTaKrpHVUZdgZeJW;
    IPlugin* mJnXzwDFPTieqFtWcZIG;

private:
    long int WbTBQxsNsCURmwRhNTAD;
    long int XGQjNlvPuckcHnviTrkP;
    long int WawamKKnqecNqBXIyHIl;
    long int YmfPcXPXNFZDznkzKZrl;
    long int YMNbgnUYZspjMLjwcIOS;
    long int YDoginwuwFxabuYCVqpT;

    long int reGtUwUlPSwEenEBVIzH;
    long int hqbKXLMjsDxRQqyJEgbg;

private:

    IScaleLayer* qJWXFXvcpbSwehmlTNru;

    float* sCDdEyIOjXBVHhcakBhd;
    float* jLmklYtHcmTxayQTpmRw;
    
    Weights qeQuIDaHqnxGPDbPoQJF;
    Weights suFVgcuEVpCOrewbJfkB;
    Weights pKmXpiCPxZwpmXlulovZ;

    void setLayerProperties();
    void loadScaleAndOffset(const char *, const char*);


};



#endif
