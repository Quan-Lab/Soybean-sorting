%%Function: Automatically detect image input size when using GPU Coder
clc 
close all 

im = imread('testbean.jpg'); %Read test image
im = imresize(im,[224,224]); %Convert image to network input size
imshow(im) %%Display image

classIdx = myModelGPU(im); %%Load forecast

load classNames 
className = classNames(classIdx); %%Output classification label
disp(['Image classified as: ' className])
disp(['Pr ' num2str(classIdx)]);
