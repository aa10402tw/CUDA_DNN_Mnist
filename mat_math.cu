#include "mat.h"
#include <iostream>

#define TILE_WIDTH 32
#define threadsPerBlock 32

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

__global__ void cuda_MatAdd(const float* A, const float* B, float* C, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n) {
        C[tid] =  __fadd_rd(A[tid], B[tid]);
        tid += stride;
    }
}

__global__ void cuda_MatSub(const float* A, const float* B, float* C, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n) {
        C[tid] =  __fsub_rd(A[tid], B[tid]);
        tid += stride;
    }
}

__global__ void cuda_MatExp(const float* src, float* dst, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n) {
        dst[tid] =  __expf(src[tid]);
        tid += stride;
    }
}

/* Element-wise multiply: C = A dot B*/
__global__ void cuda_MatEleMul(const float* A, const float* B, float* C, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n) {
        C[tid] = __fmul_rd(A[tid], B[tid]);
        tid += stride;
    }
}

/* Element-wise multiply: B = m x A*/
__global__ void cuda_MatEleMul(const float* A, const float m, float* B, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n) {
        B[tid] = __fmul_rd(A[tid], m);
        tid += stride;
    }
}

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

// Sigmoid(x) = 1 / ( 1 + exp(-x) )
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

// dSigmoid(x) = exp(-x) / (1+exp(-x))^2
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



gpuMat* MatMul(const gpuMat* A, const gpuMat* B) {
    if (A->Data == nullptr || B->Data == nullptr || A->cols != B->rows) {
        std::cout << "Invalid matmul" << std::endl;
        exit(0);
    }
    gpuMat* C = new gpuMat(A->rows, B->cols, A->channels);
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

gpuMat* MatAdd(const gpuMat* A, const gpuMat* B) {
    gpuMat* C = new gpuMat(A->rows, A->cols, A->channels);
    int size = A->getSize();
    size_t block_size = threadsPerBlock;
    size_t num_blocks = (size / block_size) + ((size % block_size) ? 1 : 0);
    cuda_MatAdd <<<num_blocks, block_size>>> (A->Data, B->Data, C->Data, size);
    return C;
}

gpuMat* MatSub(const gpuMat* A, const gpuMat* B) {
    gpuMat* C = new gpuMat(A->rows, A->cols, A->channels);
    int size = A->getSize();
    size_t block_size = threadsPerBlock;
    size_t num_blocks = (size / block_size) + ((size % block_size) ? 1 : 0);
    cuda_MatSub <<<num_blocks, block_size>>> (A->Data, B->Data, C->Data, size);
    return C;
}

gpuMat* MatExp(const gpuMat* A) {
    gpuMat* B = new gpuMat(A->rows, A->cols, A->channels);
    int size = A->getSize();
    size_t block_size = threadsPerBlock;
    size_t num_blocks = (size / block_size) + ((size % block_size) ? 1 : 0);
    cuda_MatExp <<<num_blocks, block_size>>> (A->Data, B->Data, size);
    return B;
}

gpuMat* MatEleMul(const gpuMat* A, const gpuMat* B) {
    gpuMat* C = new gpuMat(A->rows, A->cols, A->channels);
    int size = A->getSize();
    size_t block_size = threadsPerBlock;
    size_t num_blocks = (size / block_size) + ((size % block_size) ? 1 : 0);
    cuda_MatEleMul <<<num_blocks, block_size>>> (A->Data, B->Data, C->Data, size);
    return C;
}

gpuMat* MatEleMul(const gpuMat* A, float m) {
    gpuMat* B = new gpuMat(A->rows, A->cols, A->channels);
    int size = A->getSize();
    size_t block_size = threadsPerBlock;
    size_t num_blocks = (size / block_size) + ((size % block_size) ? 1 : 0);
    cuda_MatEleMul <<<num_blocks, block_size>>> (A->Data, m, B->Data, size);
    return B;

}

gpuMat* Transpose(const gpuMat* A) {
    if (A->Data == nullptr) {
        std::cout << "Error for transpose" << std::endl;
        exit(0);
    }
    gpuMat *B = new gpuMat(A->cols, A->rows, A->channels);
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

gpuMat* ReLU(const gpuMat* A) {
    if (A->Data == nullptr) {
        std::cout << "Error for ReLU" << std::endl;
        exit(0);
    }
    gpuMat *B = new gpuMat(A->rows, A->cols, A->channels);
    int size = A->getSize();
    size_t block_size = threadsPerBlock;
    size_t num_blocks = (size / block_size) + ((size % block_size) ? 1 : 0);
    cuda_ReLU <<<num_blocks, block_size>>> (A->Data, B->Data, size);
    return B;
}

gpuMat* dReLU(const gpuMat* A) {
    if (A->Data == nullptr) {
        std::cout << "Error for dReLU" << std::endl;
        exit(0);
    }
    gpuMat *B = new gpuMat(A->rows, A->cols, A->channels);
    int size = A->getSize();
    size_t block_size = threadsPerBlock;
    size_t num_blocks = (size / block_size) + ((size % block_size) ? 1 : 0);
    cuda_dReLU <<<num_blocks, block_size>>> (A->Data, B->Data, size);
    return B;
}

gpuMat* Sigmoid(const gpuMat* A) {
    if (A->Data == nullptr) {
        std::cout << "Error for Sigmoid" << std::endl;
        exit(0);
    }
    gpuMat *B = new gpuMat(A->rows, A->cols, A->channels);
    int size = A->getSize();
    size_t block_size = threadsPerBlock;
    size_t num_blocks = (size / block_size) + ((size % block_size) ? 1 : 0);
    cuda_Sigmoid <<<num_blocks, block_size>>> (A->Data, B->Data, size);
    return B;
}

gpuMat* dSigmoid(const gpuMat* A) {
    if (A->Data == nullptr) {
        std::cout << "Error for Sigmoid" << std::endl;
        exit(0);
    }
    gpuMat *B = new gpuMat(A->rows, A->cols, A->channels);
    int size = A->getSize();
    size_t block_size = threadsPerBlock;
    size_t num_blocks = (size / block_size) + ((size % block_size) ? 1 : 0);
    cuda_dSigmoid <<<num_blocks, block_size>>> (A->Data, B->Data, size);
    return B;
}


gpuMat* softmax(const gpuMat *input) {
    gpuMat* tmp = MatExp(input);
    cpuMat* cm = new cpuMat(*tmp);
    float sum = 0.0;
    for (int idx=0; idx<cm->getSize(); idx++) {
        sum += cm->Data[idx];
    }
    float m = 1 / sum;
    gpuMat* output = MatEleMul(tmp, m);
    return output;
}

