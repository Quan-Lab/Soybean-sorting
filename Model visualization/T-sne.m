% % Function: T-SNE visualization of network model
net = mobilenet-improved;
imds = imageDatastore('Dataset', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
[XTrain,XValidation] = splitEachLabel(imds,1,'randomized');

% % % Classify Validation Data
figure();
YPred = classify(net,XValidation);
confusionchart(XValidation.Labels,YPred,'ColumnSummary',"column-normalized")

% % % Compute Activations for Several Layers 
% The first layer of pooling
% The last layer of convolution
% softmax

earlyLayerName = "The first layer of pooling";
finalConvLayerName = "The last layer of convolution";
softmaxLayerName = "Softmax";
pool1Activations = activations(net,...
    XValidation,earlyLayerName,"OutputAs","rows");
finalConvActivations = activations(net,...
    XValidation,finalConvLayerName,"OutputAs","rows");
softmaxActivations = activations(net,...
    XValidation,softmaxLayerName,"OutputAs","rows");

% % % Ambiguity of Classifications
[R,RI] = maxk(softmaxActivations,2,2);
ambiguity = R(:,2)./R(:,1);
[ambiguity,ambiguityIdx] = sort(ambiguity,"descend");
classList = unique(XValidation.Labels);
top10Idx = ambiguityIdx(1:10);
top10Ambiguity = ambiguity(1:10);
mostLikely = classList(RI(ambiguityIdx,1));
secondLikely = classList(RI(ambiguityIdx,2));
table(top10Idx,top10Ambiguity,mostLikely(1:10),secondLikely(1:10),XValidation.Labels(ambiguityIdx(1:10)),...
    'VariableNames',["Image #","Ambiguity","Likeliest","Second","True Class"])

v = 27;
figure();
imshow(XValidation.Files{v});
title(sprintf("Observation: %i\n" + ...
    "Actual: %s. Predicted: %s", v, ...
    string(XValidation.Labels(v)), string(YPred(v))), ...
    'Interpreter', 'none');

% % % Compute 2-D Representations of Data Using t-SNE
rng default
pool1tsne = tsne(pool1Activations);
finalConvtsne = tsne(finalConvActivations);
softmaxtsne = tsne(softmaxActivations);

% % % Compare Network Behavior for Early and Later Layers
doLegend = 'off';
Legend = 'on';
markerSize = 7;
figure;

subplot(1,3,1);
gscatter(pool1tsne(:,1),pool1tsne(:,2),XValidation.Labels, ...
    [],'.',markerSize,Legend);
title("Max pooling activations");

subplot(1,3,2);
gscatter(finalConvtsne(:,1),finalConvtsne(:,2),XValidation.Labels, ...
    [],'.',markerSize,doLegend);
title("Final conv activations");

subplot(1,3,3);
gscatter(softmaxtsne(:,1),softmaxtsne(:,2),XValidation.Labels, ...
    [],'.',markerSize,doLegend);
title("Softmax activations");
