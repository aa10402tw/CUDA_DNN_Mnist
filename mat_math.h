#ifndef MAT_MATH_H
#define MAT_MATH_H


__global__ void cuda_MatMul(const float* A, const float* B, float* C,
                       int ARows, int ACols,
                       int BRows, int BCols,
                       int CRows, int CCols);

__global__ void cuda_MatAdd(const float* A, const float* B, float* C, int n);

__global__ void cuda_MatSub(const float* A, const float* B, float* C, int n);

__global__ void cuda_MatExp(const float* src, float* dst, int n);

__global__ void cuda_Transpose(const float* src, float* dst, int srcCols, int dstCols, int n);

__global__ void cuda_MatEleMul(const float* A, const float* B, float* C, int n);

__global__ void cuda_MatEleMul(const float* A, const float m, float* B, int n);

__global__ void cuda_ReLU(const float* src, float* dst, int n);

__global__ void cuda_dReLU(const float* src, float* dst, int n);

__global__ void cuda_Sigmoid(const float* src, float* dst, int n);

__global__ void cuda_dSigmoid(const float* src, float* dst, int n);


gpuMat* MatMul(const gpuMat* A, const gpuMat* B);

gpuMat* MatAdd(const gpuMat* A, const gpuMat* B);

gpuMat* MatSub(const gpuMat* A, const gpuMat* B);

gpuMat* MatExp(const gpuMat* A);

gpuMat* MatEleMul(const gpuMat* A, const gpuMat* B);

gpuMat* MatEleMul(const gpuMat* A, float m);

gpuMat* Transpose(const gpuMat* A);


gpuMat* Sigmoid(const gpuMat* A);

gpuMat* dSigmoid(const gpuMat* A);

gpuMat* ReLU(const gpuMat* A);

gpuMat* dReLU(const gpuMat* A);

gpuMat* softmax(const gpuMat *input);

#endif // MAT_MATH_H
