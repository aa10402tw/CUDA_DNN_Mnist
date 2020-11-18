#include "mat.h"
#include <iostream>

#define TILE_WIDTH 32
#define threadsPerBlock 32

// Compute matrix multiplication C = A x B
__global__ void cuda_MatMul(const float* A, const float* B, float* C,
                       int ARows, int ACols,
                       int BRows, int BCols,
                       int CRows, int CCols)
{
    float CValue = 0;

    int Row = blockIdx.y*TILE_WIDTH + threadIdx.y;
    int Col = blockIdx.x*TILE_WIDTH + threadIdx.x;

    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    for (int k = 0; k < (TILE_WIDTH + ACols - 1)/TILE_WIDTH; k++) {
        /* Allocate corresponding value to As and Bs */
         if (k*TILE_WIDTH + threadIdx.x < ACols && Row < ARows)
             As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_WIDTH + threadIdx.x];
         else
             As[threadIdx.y][threadIdx.x] = 0.0;

         if (k*TILE_WIDTH + threadIdx.y < BRows && Col < BCols)
             Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_WIDTH+ threadIdx.y)*BCols + Col];
         else
             Bs[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

        /* Compute Value of block sum */
         for (int n = 0; n < TILE_WIDTH; ++n)
             CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

         __syncthreads();
    }

    if (Row < CRows && Col < CCols)
        C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
}

// Compute matrix Addition C = A + B
__global__ void cuda_MatAdd(const float* A, const float* B, float* C, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n) {
        C[tid] =  __fadd_rd(A[tid], B[tid]);
        tid += stride;
    }
}

// Compute matrix Subtraction C = A - B
__global__ void cuda_MatSub(const float* A, const float* B, float* C, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n) {
        C[tid] =  __fsub_rd(A[tid], B[tid]);
        tid += stride;
    }
}

// Element-Wise Multiply C[i][j] = A[i][j] * B[i][j]
__global__ void cuda_MatEleMul(const float* A, const float* B, float* C, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n) {
        C[tid] = __fmul_rd(A[tid], B[tid]);
        tid += stride;
    }
}

// Element-Wise Multiply C[i][j] = A[i][j] * m
__global__ void cuda_MatEleMul(const float* A, const float m, float* B, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n) {
        B[tid] = __fmul_rd(A[tid], m);
        tid += stride;
    }
}

// Matrix Transpose B = A^T
__global__ void cuda_Transpose(const float* src, float* dst, int srcCols, int dstCols, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n) {
        int c_dst = tid % dstCols;
        int r_dst = tid / dstCols;
        int r_src = c_dst;
        int c_src = r_dst;
        dst[tid] = src[r_src * srcCols + c_src];
        tid += stride;
    }
}

// Compute matrix exponential operation B[i][j] = exp(A[i][j])
__global__ void cuda_MatExp(const float* src, float* dst, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n) {
        dst[tid] =  __expf(src[tid]);
        tid += stride;
    }
}

// Compute matrix ReLU operation B[i][j] = ReLU(A[i][j]) [B[i][j] = max(0, A[i][j])]
__global__ void cuda_ReLU(const float* src, float* dst, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n) {
        if (src[tid] > 0.0)
            dst[tid] = src[tid];
        else
            dst[tid] = 0.0;
        tid += stride;
    }
}

// Compute matrix derivative of ReLU operation B[i][j] = dReLU(A[i][j]) [B[i][j] = 1 if A[i][j] > 0 else 0]
__global__ void cuda_dReLU(const float* src, float* dst, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n) {
        if (src[tid] > 0.0)
            dst[tid] = 1.0;
        else
            dst[tid] = 0.0;
        tid += stride;
    }
}

// Compute matrix Sigmoid operation B[i][j] = Sigmoid(A[i][j])  [S(x) = 1 / (1+exp(-x)) ]
__global__ void cuda_Sigmoid(const float* src, float* dst, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n) {
        float v = __fmul_rd(src[tid], -1.0);    // -x
		v = __expf(v);                          // exp(-x)
		v = __fadd_rd(v, 1.0);                  // 1 + exp(-x)
		dst[tid] = __fdividef(1.0, v);          // 1 / ( 1 + exp(-x) )
		tid += stride;
    }

}

// Compute matrix derivative of Sigmoid function  B[i][j] = dSigmoid(A[i][j]) [dSigmoid(x) = exp(-x) / (1+exp(-x))^2]
__global__ void cuda_dSigmoid(const float* src, float* dst, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n) {
        float v = __fmul_rd(src[tid], -1.0);    // -x
        v = __expf(src[tid]);                   // exp(-x)
		float v2 = __fadd_rd(v, 1.0);           // exp(-x)+1
		v2 = __fmul_rd(v2, v2);                 // (exp(-x)+1)^2
		dst[tid] = fdividef(v, v2);             // exp(-x) / (1+exp(-x))^2
		tid += stride;
    }
}



GpuMat* MatMul(const GpuMat* A, const GpuMat* B) {
    if (A->Data == nullptr || B->Data == nullptr || A->cols != B->rows) {
        std::cout << "Invalid Matrix Multiplication" << std::endl;
        exit(0);
    }
    GpuMat* C = new GpuMat(A->rows, B->cols, A->channels);
    int sizeA = A->rows * A->cols;
    int sizeB = B->rows * B->cols;
    int sizeC = C->rows * C->cols;

    dim3 dimGrid( (C->cols-1)/TILE_WIDTH + 1, (C->rows - 1) / TILE_WIDTH + 1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    for (int c=0; c<A->channels; c++) {
        int offsetA = c*sizeA;
        int offsetB = c*sizeB;
        int offsetC = c*sizeC;
        cuda_MatMul<<<dimGrid, dimBlock>>>(A->Data+offsetA, B->Data+offsetB, C->Data+offsetC,
                                           A->rows, A->cols,
                                           B->rows, B->cols,
                                           C->rows, C->cols);
    }
    return C;
}

GpuMat* MatAdd(const GpuMat* A, const GpuMat* B) {
    if (A->Data == nullptr || B->Data == nullptr || 
        !(A->rows == B->rows && A->cols == B->cols && A->channels == B->channels)) {
        std::cout << "Invalid Matrix Addition" << std::endl;
        exit(0);
    }
    GpuMat* C = new GpuMat(A->rows, A->cols, A->channels);
    int size = A->getSize();
    size_t block_size = threadsPerBlock;
    size_t num_blocks = (size / block_size) + ((size % block_size) ? 1 : 0);
    cuda_MatAdd <<<num_blocks, block_size>>> (A->Data, B->Data, C->Data, size);
    return C;
}

GpuMat* MatSub(const GpuMat* A, const GpuMat* B) {
    if (A->Data == nullptr || B->Data == nullptr || 
        !(A->rows == B->rows && A->cols == B->cols && A->channels == B->channels)) {
        std::cout << "Invalid Matrix Subtraction" << std::endl;
        exit(0);
    }
    GpuMat* C = new GpuMat(A->rows, A->cols, A->channels);
    int size = A->getSize();
    size_t block_size = threadsPerBlock;
    size_t num_blocks = (size / block_size) + ((size % block_size) ? 1 : 0);
    cuda_MatSub <<<num_blocks, block_size>>> (A->Data, B->Data, C->Data, size);
    return C;
}

GpuMat* MatEleMul(const GpuMat* A, const GpuMat* B) {
    if (A->Data == nullptr || B->Data == nullptr || 
        !(A->rows == B->rows && A->cols == B->cols && A->channels == B->channels)) {
        std::cout << "Invalid Matrix Elementwise Multiplication" << std::endl;
        exit(0);
    }
    GpuMat* C = new GpuMat(A->rows, A->cols, A->channels);
    int size = A->getSize();
    size_t block_size = threadsPerBlock;
    size_t num_blocks = (size / block_size) + ((size % block_size) ? 1 : 0);
    cuda_MatEleMul <<<num_blocks, block_size>>> (A->Data, B->Data, C->Data, size);
    return C;
}

GpuMat* MatEleMul(const GpuMat* A, float m) {
    if (A->Data == nullptr) {
        std::cout << "Invalid Matrix Elementwise Multiplication" << std::endl;
        exit(0);
    }
    GpuMat* B = new GpuMat(A->rows, A->cols, A->channels);
    int size = A->getSize();
    size_t block_size = threadsPerBlock;
    size_t num_blocks = (size / block_size) + ((size % block_size) ? 1 : 0);
    cuda_MatEleMul <<<num_blocks, block_size>>> (A->Data, m, B->Data, size);
    return B;

}

GpuMat* MatEleDiv(const GpuMat* A, float m) {
    return MatEleMul(A, 1/m);
}

GpuMat* Transpose(const GpuMat* A) {
    if (A->Data == nullptr) {
        std::cout << "Invalid Matrix Transpose" << std::endl;
        exit(0);
    }
    GpuMat *B = new GpuMat(A->cols, A->rows, A->channels);
    int size = B->rows * B->cols;
    size_t block_size = threadsPerBlock;
    size_t num_blocks = (size / block_size) + ((size % block_size) ? 1 : 0);
    for (int c=0; c<A->channels; c++) {
        int offset = c*size;
        cuda_Transpose <<<num_blocks, block_size>>> (A->Data+offset, B->Data+offset,
                                                     A->cols, B->cols, size);
    }
    return B;
}

GpuMat* MatExp(const GpuMat* A) {
    if (A->Data == nullptr) {
        std::cout << "Invalid Matrix Exponential" << std::endl;
        exit(0);
    }
    GpuMat* B = new GpuMat(A->rows, A->cols, A->channels);
    int size = A->getSize();
    size_t block_size = threadsPerBlock;
    size_t num_blocks = (size / block_size) + ((size % block_size) ? 1 : 0);
    cuda_MatExp <<<num_blocks, block_size>>> (A->Data, B->Data, size);
    return B;
}

GpuMat* ReLU(const GpuMat* A) {
    if (A->Data == nullptr) {
        std::cout << "Invalid Matrix ReLU" << std::endl;
        exit(0);
    }
    GpuMat *B = new GpuMat(A->rows, A->cols, A->channels);
    int size = A->getSize();
    size_t block_size = threadsPerBlock;
    size_t num_blocks = (size / block_size) + ((size % block_size) ? 1 : 0);
    cuda_ReLU <<<num_blocks, block_size>>> (A->Data, B->Data, size);
    return B;
}

GpuMat* dReLU(const GpuMat* A) {
    if (A->Data == nullptr) {
        std::cout << "Invalid Matrix dReLU" << std::endl;
        exit(0);
    }
    GpuMat *B = new GpuMat(A->rows, A->cols, A->channels);
    int size = A->getSize();
    size_t block_size = threadsPerBlock;
    size_t num_blocks = (size / block_size) + ((size % block_size) ? 1 : 0);
    cuda_dReLU <<<num_blocks, block_size>>> (A->Data, B->Data, size);
    return B;
}

GpuMat* Sigmoid(const GpuMat* A) {
    if (A->Data == nullptr) {
        std::cout << "Invalid Matrix Sigmoid" << std::endl;
        exit(0);
    }
    GpuMat *B = new GpuMat(A->rows, A->cols, A->channels);
    int size = A->getSize();
    size_t block_size = threadsPerBlock;
    size_t num_blocks = (size / block_size) + ((size % block_size) ? 1 : 0);
    cuda_Sigmoid <<<num_blocks, block_size>>> (A->Data, B->Data, size);
    return B;
}

GpuMat* dSigmoid(const GpuMat* A) {
    if (A->Data == nullptr) {
        std::cout << "Invalid Matrix dSigmoid" << std::endl;
        exit(0);
    }
    GpuMat *B = new GpuMat(A->rows, A->cols, A->channels);
    int size = A->getSize();
    size_t block_size = threadsPerBlock;
    size_t num_blocks = (size / block_size) + ((size % block_size) ? 1 : 0);
    cuda_dSigmoid <<<num_blocks, block_size>>> (A->Data, B->Data, size);
    return B;
}

GpuMat* softmax(const GpuMat *input) {
    GpuMat* tmp = MatExp(input);
    CpuMat* cm = CpuMat::copy(tmp);
    float sum = 0.0;
    for (int idx=0; idx<cm->getSize(); idx++) {
        sum += cm->Data[idx];
    }
    float m = 1 / sum;
    GpuMat* output = MatEleMul(tmp, m);

    delete tmp;
    delete cm;
    return output;
}

