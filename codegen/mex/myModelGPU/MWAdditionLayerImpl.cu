#include "MWAdditionLayer.hpp"
#include "MWAdditionLayerImpl.hpp"
 MWAdditionLayerImpl::MWAdditionLayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl, int )  : MWCNNLayerImpl(layer, ntwk_impl)  { 
ITensor* prevLayerTensor1 = getInputITensor(0); ITensor* prevLayerTensor2 = 
getInputITensor(1); ABtNoHVrQOgivJIJagNR = 
eWYFXrUazhqiEIscccda->network->addElementWise(*prevLayerTensor1, 
*prevLayerTensor2, ElementWiseOperation::kSUM); 
ABtNoHVrQOgivJIJagNR->setName(getLayer()->getName().c_str()); 
for (int i = 2; i < getLayer()->getNumInputs(); ++i){ 
ABtNoHVrQOgivJIJagNR = 
eWYFXrUazhqiEIscccda->network->addElementWise(*ABtNoHVrQOgivJIJagNR->getOutput(0), 
*getInputITensor(i), ElementWiseOperation::kSUM);  } 
setOpTensorPtr(ABtNoHVrQOgivJIJagNR->getOutput(0)); return ; } 
MWAdditionLayerImpl::~MWAdditionLayerImpl() { }