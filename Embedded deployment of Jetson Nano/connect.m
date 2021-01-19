%%Function: Used to connect embedded devices and verify the environment
mex -setup %Choose C language to compile
hwobj = jetson('host-name','username','password'); %Connect to NVIDIA Jetson with username and password
envCfg = coder.gpuEnvConfig('jetson'); %Use the coder.checkGpuInstall function to verify that the required compilers and libraries are set up correctly
envCfg.DeepLibTarget = 'cudnn' ; 
envCfg.DeepCodegen = 1; %Check the required compilers and libraries
envCfg.Quiet = 1;
envCfg.HardwareObject = hwobj;
coder.checkGpuInstall(envCfg); 
cfg = coder.gpuConfig('exe'); %Create a GPU encoder configuration object for generating executable files
cfg.Hardware = coder.hardware('NVIDIA Jetson'); %Jetson platform creates configuration objects
cfg.DeepLearningConfig = coder.DeepLearningConfig('cudnn'); %Set the deep learning configuration to 'cudnn'
cfg.CustomSource=fullfile('main_MobileNet-improve.h'); %Custom wrapper to call forecast function in generated code
cfg.CustomSource=fullfile('main_MobileNet-improve.cu'); 
codegen -config cfg -args {im} myModelGPU -report %Generate code on the host