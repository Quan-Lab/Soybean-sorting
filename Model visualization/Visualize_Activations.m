% % Function: Visualize the activation of a specific layer in the network model

% Load the network model
net = mobilenet-improved;

% Read and display image
im = imread('soybean.jpg');
imshow(im)
imgSize = size(im);
imgSize = imgSize(1:2);

% Analyze the network to see which layers can be viewed
analyzeNetwork(net)

% Show the activation of the first convolutional layer
act1 = activations(net,im,'conv1');
sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
I = imtile(mat2gray(act1),'GridSize',[8 8]);
imshow(I)

% Show the activation of the last convolutional layer
act1 = activations(net,im,'Conv_1');
sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
I = imtile(mat2gray(act1),'GridSize',[8 8]);
imshow(I)


