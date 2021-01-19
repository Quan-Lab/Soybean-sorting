%%Function: Used to generate C language deployment files
function classIdx = myModelGPU(im)
%%Load optimization model
persistent net
if isempty(net)
    net = coder.loadDeepLearningNetwork('mobilenet-improved.mat');
end
output = predict(net,im);%%Output result
[~,classIdx] = max(output);

end

