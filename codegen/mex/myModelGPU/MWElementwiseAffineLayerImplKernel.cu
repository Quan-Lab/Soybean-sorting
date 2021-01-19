#include "MWKernelHeaders.hpp"
#include <math.h>
#include <stdio.h>
 void __global__ __launch_bounds__(1024) scale_scalar_kernel(float* 
inputBuffer, float* outputBuffer, float* omxlPZbBePZdWaJOBUUG, long int 
YNmJhGSUszJKxsodxiuV) {  for (int idx = blockDim.x * blockIdx.x + threadIdx.x; 
idx < YNmJhGSUszJKxsodxiuV; idx += blockDim.x * gridDim.x) {  outputBuffer[idx] 
= omxlPZbBePZdWaJOBUUG[0]*inputBuffer[idx]; } } void __global__ 
__launch_bounds__(1024) scale_vector_kernel(float* inputBuffer, float* 
outputBuffer, float* omxlPZbBePZdWaJOBUUG, long int YeIFysyIXePEVfpcANol, 
long int YOWMnLKOMqAODXiVNoGy, long int YNmJhGSUszJKxsodxiuV) {  for 
(int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < YNmJhGSUszJKxsodxiuV; 
idx += blockDim.x * gridDim.x) { int dMxIKDGTITyhdLqIHBLA = 
idx/YOWMnLKOMqAODXiVNoGy; long int FLuSVNoPhAFKtLUchSvv = 
idx-(YOWMnLKOMqAODXiVNoGy*dMxIKDGTITyhdLqIHBLA); int LgxABSJPBXdCozJkFqTg = 
static_cast<int>(FLuSVNoPhAFKtLUchSvv / YeIFysyIXePEVfpcANol); 
outputBuffer[idx] = omxlPZbBePZdWaJOBUUG[LgxABSJPBXdCozJkFqTg]*inputBuffer[idx]; } } void 
__global__ __launch_bounds__(1024) scale_matrix2d_kernel(float* inputBuffer, 
float* outputBuffer, float* omxlPZbBePZdWaJOBUUG, long int YgcpEBUCwCLaPhyntIio, long 
int YeIFysyIXePEVfpcANol, long int YOWMnLKOMqAODXiVNoGy, long 
int YNmJhGSUszJKxsodxiuV) {  for (int idx = blockDim.x * blockIdx.x + 
threadIdx.x; idx < YNmJhGSUszJKxsodxiuV; idx += blockDim.x * gridDim.x) { int 
dMxIKDGTITyhdLqIHBLA = idx/YOWMnLKOMqAODXiVNoGy; long int FLuSVNoPhAFKtLUchSvv 
= idx-(YOWMnLKOMqAODXiVNoGy*dMxIKDGTITyhdLqIHBLA); int LgxABSJPBXdCozJkFqTg = 
static_cast<int>(FLuSVNoPhAFKtLUchSvv / YeIFysyIXePEVfpcANol); long 
int FOcStuqCptsGIZXskVpC = FLuSVNoPhAFKtLUchSvv - 
(YeIFysyIXePEVfpcANol*LgxABSJPBXdCozJkFqTg); int vIWQzNvYZSuxmOTVDFhU = 
static_cast<int>(FOcStuqCptsGIZXskVpC % YgcpEBUCwCLaPhyntIio); int 
RVrPByQXdKmunRZHKWJD = static_cast<int>(FOcStuqCptsGIZXskVpC / YgcpEBUCwCLaPhyntIio); 
outputBuffer[idx] = 
omxlPZbBePZdWaJOBUUG[vIWQzNvYZSuxmOTVDFhU+YgcpEBUCwCLaPhyntIio*RVrPByQXdKmunRZHKWJD]*inputBuffer[idx]; 
} } void __global__ __launch_bounds__(1024) scale_tensor3d_kernel(float* 
inputBuffer, float* outputBuffer, float* omxlPZbBePZdWaJOBUUG, long int 
YgcpEBUCwCLaPhyntIio, long int YGiQICncmsGZkNUyiQyg, long int 
YeIFysyIXePEVfpcANol, long int YOWMnLKOMqAODXiVNoGy, long int 
YNmJhGSUszJKxsodxiuV) {  for (int idx = blockDim.x * blockIdx.x + threadIdx.x; 
idx < YNmJhGSUszJKxsodxiuV; idx += blockDim.x * gridDim.x) { int dMxIKDGTITyhdLqIHBLA = 
idx/YOWMnLKOMqAODXiVNoGy; long int FLuSVNoPhAFKtLUchSvv = 
idx-(YOWMnLKOMqAODXiVNoGy*dMxIKDGTITyhdLqIHBLA); int LgxABSJPBXdCozJkFqTg = 
static_cast<int>(FLuSVNoPhAFKtLUchSvv / YeIFysyIXePEVfpcANol); long 
int FOcStuqCptsGIZXskVpC = FLuSVNoPhAFKtLUchSvv - 
(YeIFysyIXePEVfpcANol*LgxABSJPBXdCozJkFqTg); int vIWQzNvYZSuxmOTVDFhU = 
static_cast<int>(FOcStuqCptsGIZXskVpC % YgcpEBUCwCLaPhyntIio); int 
RVrPByQXdKmunRZHKWJD = static_cast<int>(FOcStuqCptsGIZXskVpC / YgcpEBUCwCLaPhyntIio); 
outputBuffer[idx] = 
omxlPZbBePZdWaJOBUUG[vIWQzNvYZSuxmOTVDFhU+YgcpEBUCwCLaPhyntIio*(RVrPByQXdKmunRZHKWJD+YGiQICncmsGZkNUyiQyg*LgxABSJPBXdCozJkFqTg)]*inputBuffer[idx]; 
} }  void __global__ __launch_bounds__(1024) offset_scalar_kernel(float* 
inputBuffer, float* outputBuffer, float* gTcJMwtYuwiqqUmqvKhT, long int 
YNmJhGSUszJKxsodxiuV, bool ZinudJuZuGitiNTsJpBR, int bUVPfnrJhLfHzOLUUrKk, int 
unSXtdjDjpysqxmbIiPv) {  for (int idx = blockDim.x * blockIdx.x + threadIdx.x; 
idx < YNmJhGSUszJKxsodxiuV; idx += blockDim.x * gridDim.x) { outputBuffer[idx] 
= inputBuffer[idx] + gTcJMwtYuwiqqUmqvKhT[0]; if (ZinudJuZuGitiNTsJpBR){ 
outputBuffer[idx] = outputBuffer[idx] > unSXtdjDjpysqxmbIiPv ? 
unSXtdjDjpysqxmbIiPv : outputBuffer[idx]; outputBuffer[idx] = 
outputBuffer[idx] < bUVPfnrJhLfHzOLUUrKk ? bUVPfnrJhLfHzOLUUrKk : 
outputBuffer[idx]; } } } void __global__ __launch_bounds__(1024) 
offset_vector_kernel(float* inputBuffer, float* outputBuffer, float* 
gTcJMwtYuwiqqUmqvKhT,  long int YeIFysyIXePEVfpcANol, long int 
YOWMnLKOMqAODXiVNoGy, long int YNmJhGSUszJKxsodxiuV, bool 
ZinudJuZuGitiNTsJpBR, int bUVPfnrJhLfHzOLUUrKk, int unSXtdjDjpysqxmbIiPv) {  
for (int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < 
YNmJhGSUszJKxsodxiuV; idx += blockDim.x * gridDim.x) { int dMxIKDGTITyhdLqIHBLA = 
idx/YOWMnLKOMqAODXiVNoGy; long int FLuSVNoPhAFKtLUchSvv = 
idx-(YOWMnLKOMqAODXiVNoGy*dMxIKDGTITyhdLqIHBLA); int LgxABSJPBXdCozJkFqTg = 
static_cast<int>(FLuSVNoPhAFKtLUchSvv / YeIFysyIXePEVfpcANol); 
outputBuffer[idx] = inputBuffer[idx] + gTcJMwtYuwiqqUmqvKhT[LgxABSJPBXdCozJkFqTg]; if 
(ZinudJuZuGitiNTsJpBR){ outputBuffer[idx] = outputBuffer[idx] > 
unSXtdjDjpysqxmbIiPv ? unSXtdjDjpysqxmbIiPv : outputBuffer[idx]; 
outputBuffer[idx] = outputBuffer[idx] < bUVPfnrJhLfHzOLUUrKk ? 
bUVPfnrJhLfHzOLUUrKk : outputBuffer[idx]; } } } void __global__ 
__launch_bounds__(1024) offset_matrix2d_kernel(float* inputBuffer, float* 
outputBuffer, float* gTcJMwtYuwiqqUmqvKhT, long int YgcpEBUCwCLaPhyntIio, long int 
YeIFysyIXePEVfpcANol, long int YOWMnLKOMqAODXiVNoGy, long int 
YNmJhGSUszJKxsodxiuV, bool ZinudJuZuGitiNTsJpBR, int bUVPfnrJhLfHzOLUUrKk, int 
unSXtdjDjpysqxmbIiPv) {  for (int idx = blockDim.x * blockIdx.x + threadIdx.x; 
idx < YNmJhGSUszJKxsodxiuV; idx += blockDim.x * gridDim.x) { int dMxIKDGTITyhdLqIHBLA = 
idx/YOWMnLKOMqAODXiVNoGy; long int FLuSVNoPhAFKtLUchSvv = 
idx-(YOWMnLKOMqAODXiVNoGy*dMxIKDGTITyhdLqIHBLA); int LgxABSJPBXdCozJkFqTg = 
static_cast<int>(FLuSVNoPhAFKtLUchSvv / YeIFysyIXePEVfpcANol); long 
int FOcStuqCptsGIZXskVpC = FLuSVNoPhAFKtLUchSvv - 
(YeIFysyIXePEVfpcANol*LgxABSJPBXdCozJkFqTg); int vIWQzNvYZSuxmOTVDFhU = 
static_cast<int>(FOcStuqCptsGIZXskVpC % YgcpEBUCwCLaPhyntIio); int 
RVrPByQXdKmunRZHKWJD = static_cast<int>(FOcStuqCptsGIZXskVpC / YgcpEBUCwCLaPhyntIio); 
outputBuffer[idx] = inputBuffer[idx] + 
gTcJMwtYuwiqqUmqvKhT[vIWQzNvYZSuxmOTVDFhU+YgcpEBUCwCLaPhyntIio*RVrPByQXdKmunRZHKWJD]; if 
(ZinudJuZuGitiNTsJpBR){ outputBuffer[idx] = outputBuffer[idx] > 
unSXtdjDjpysqxmbIiPv ? unSXtdjDjpysqxmbIiPv : outputBuffer[idx]; 
outputBuffer[idx] = outputBuffer[idx] < bUVPfnrJhLfHzOLUUrKk ? 
bUVPfnrJhLfHzOLUUrKk : outputBuffer[idx]; } } } void __global__ 
__launch_bounds__(1024) offset_tensor3d_kernel(float* inputBuffer, float* 
outputBuffer, float* gTcJMwtYuwiqqUmqvKhT,  long int YgcpEBUCwCLaPhyntIio, long int 
YGiQICncmsGZkNUyiQyg, long int YeIFysyIXePEVfpcANol, long int 
YOWMnLKOMqAODXiVNoGy, long int YNmJhGSUszJKxsodxiuV, bool 
ZinudJuZuGitiNTsJpBR, int bUVPfnrJhLfHzOLUUrKk, int unSXtdjDjpysqxmbIiPv) {  
for (int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < 
YNmJhGSUszJKxsodxiuV; idx += blockDim.x * gridDim.x) { int dMxIKDGTITyhdLqIHBLA = 
idx/YOWMnLKOMqAODXiVNoGy; long int FLuSVNoPhAFKtLUchSvv = 
idx-(YOWMnLKOMqAODXiVNoGy*dMxIKDGTITyhdLqIHBLA); int LgxABSJPBXdCozJkFqTg = 
static_cast<int>(FLuSVNoPhAFKtLUchSvv / YeIFysyIXePEVfpcANol); long 
int FOcStuqCptsGIZXskVpC = FLuSVNoPhAFKtLUchSvv - 
(YeIFysyIXePEVfpcANol*LgxABSJPBXdCozJkFqTg); int vIWQzNvYZSuxmOTVDFhU = 
static_cast<int>(FOcStuqCptsGIZXskVpC % YgcpEBUCwCLaPhyntIio); int 
RVrPByQXdKmunRZHKWJD = static_cast<int>(FOcStuqCptsGIZXskVpC / YgcpEBUCwCLaPhyntIio); 
outputBuffer[idx] = inputBuffer[idx] + 
gTcJMwtYuwiqqUmqvKhT[vIWQzNvYZSuxmOTVDFhU+YgcpEBUCwCLaPhyntIio*(RVrPByQXdKmunRZHKWJD+YGiQICncmsGZkNUyiQyg*LgxABSJPBXdCozJkFqTg)]; 
if (ZinudJuZuGitiNTsJpBR){ outputBuffer[idx] = outputBuffer[idx] > 
unSXtdjDjpysqxmbIiPv ? unSXtdjDjpysqxmbIiPv : outputBuffer[idx]; 
outputBuffer[idx] = outputBuffer[idx] < bUVPfnrJhLfHzOLUUrKk ? 
bUVPfnrJhLfHzOLUUrKk : outputBuffer[idx]; } } } 