% % Function: Test the network model and generate a confusion matrix
load('mobilenet-improved_1.mat', 'netTransfer'); %Import network model

imds = imageDatastore('Testset', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
[XTrain,XValidation] = splitEachLabel(imds,1,'randomized');
YTrain = XTrain.Labels;
YValidation = XValidation.Labels;

YPredicted = classify(netTransfer,XValidation);

plotconfusion(YValidation,YPredicted);
saveas(1,'mob-improve_1.jpg'); %Save confusion matrix image



