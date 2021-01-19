/* Copyright 2017-2018 The MathWorks, Inc. */

						
#ifndef GPUCODER_BATCHNORMALIZATIONIMPL_HPP
#define GPUCODER_BATCHNORMALIZATIONIMPL_HPP

#include "MWCNNLayerImpl.hpp"
#include "MWTargetNetworkImpl.hpp"

/**
 * Codegen class for Batch Normalization Layer
 *
 * This layer performs a simple scale and offset of the input data
 * using previously learned weights together with measured mean and
 * variance over the training data.
 */

class MWBatchNormalizationLayerImpl : public MWCNNLayerImpl
{
  public:
    MWBatchNormalizationLayerImpl(MWCNNLayer *,
                                  MWTargetNetworkImpl *,
                                  double const,
                                  const char*,
                                  const char*,
                                  const char*,
                                  const char*,
                                  int,                                  
                                  int);
    ~MWBatchNormalizationLayerImpl();

    void cleanup();
    
  protected:
    // Methods to setup the scale, offset, mean and variance parameters
    void loadScale(const char*);
    void loadOffset(const char*);
    void loadTrainedMean(const char*);
    void loadTrainedVariance(const char*);

  private:
    double RgALmBtPIZWDevjZBUHy;

	// Parameters from training
    float* pxmnUEWGnfCxJNuDkXAo;
    float* hYTzvgWajqchLzrmxjqn;
    float* vcOGADqMrTrPPcuYvrHS;
    float* vlzcDcTSrYXZiamsmNlx;

    float* qLXeoFROCbISdsnwpYgl;
    float* sXWXkiDEKpurgeCqZLDL;
    float* niGnnRufksTFnsUUxnCj;

	// MW_MANGLED PREFIX
    IScaleLayer* ATYqlAsSnRELrakAbCoK;

    Weights qeQuIDaHqnxGPDbPoQJF;
    Weights suFVgcuEVpCOrewbJfkB;
    Weights pKmXpiCPxZwpmXlulovZ;

    bool QJJBjzDRkBQsCLkHaADa;

  private:
  	/** Helper to load a parameter from file into GPU memory. */
  	void iLoadParamOntoGPU(char const * const RuGYRQXjIMQJrbgoRUxZ,
  							int const fdiBdaeFcIDdmsgMxaJT,
                            float* EMtxAWxHxCcPIkaNDIHM);

    /** Helper to load a parameter from file into host memory. */
    void iLoadParam(char const * const RuGYRQXjIMQJrbgoRUxZ,
                            int const fdiBdaeFcIDdmsgMxaJT,
                            float* EMtxAWxHxCcPIkaNDIHM);
};

#endif
