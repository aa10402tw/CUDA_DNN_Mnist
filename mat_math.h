#include "mat.h"

#ifndef MAT_MATH_H
#define MAT_MATH_H

__global__ void cuda_MatMul(const float* A, const float* B, float* C,
                       int ARows, int ACols,
                       int BRows, int BCols,
                       int CRows, int CCols);

__global__ void cuda_MatAdd(const float* A, const float* B, float* C, int n);

__global__ void cuda_MatSub(const float* A, const float* B, float* C, int n);

__global__ void cuda_MatEleMul(const float* A, const float* B, float* C, int n);

__global__ void cuda_MatEleMul(const float* A, const float m, float* B, int n);

__global__ void cuda_Transpose(const float* src, float* dst, int srcCols, int dstCols, int n);

__global__ void cuda_MatExp(const float* src, float* dst, int n);

__global__ void cuda_ReLU(const float* src, float* dst, int n);

__global__ void cuda_dReLU(const float* src, float* dst, int n);

__global__ void cuda_Sigmoid(const float* src, float* dst, int n);

__global__ void cuda_dSigmoid(const float* src, float* dst, int n);


GpuMat* MatMul(const GpuMat* A, const GpuMat* B);

GpuMat* MatAdd(const GpuMat* A, const GpuMat* B);

GpuMat* MatSub(const GpuMat* A, const GpuMat* B);

GpuMat* MatEleMul(const GpuMat* A, const GpuMat* B);

GpuMat* MatEleMul(const GpuMat* A, float m);

GpuMat* MatEleDiv(const GpuMat* A, float m);

GpuMat* Transpose(const GpuMat* A);

GpuMat* MatExp(const GpuMat* A);

GpuMat* Sigmoid(const GpuMat* A);

GpuMat* dSigmoid(const GpuMat* A);

GpuMat* ReLU(const GpuMat* A);

GpuMat* dReLU(const GpuMat* A);

GpuMat* softmax(const GpuMat *input);

#endif // MAT_MATH_H
