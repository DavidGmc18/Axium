#include "AXM/operations.hpp"
#include "../util.hpp"

namespace axm::op::cuda {

__global__ void kernel(
    size_t M, size_t N, size_t K,
    const float alpha,
    const float* A, size_t lda, bool trana,
    const float* B, size_t ldb, bool tranb,
    const float beta,
    float* C, size_t ldc
) {
    const size_t n = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t m = size_t(blockIdx.y) * blockDim.y + threadIdx.y;

    if (m < M && n < N) {
        float sum = 0.0f;
        for (size_t k = 0; k < K; ++k) {
            float a = trana ? A[k * lda + m] : A[m * lda + k];
            float b = tranb ? B[n * ldb + k] : B[k * ldb + n];
            sum += a * b;
        }
        C[m * ldc + n] = alpha * sum + beta * C[m * ldc + n];
    }
}

template<const int BLOCK_SIZE>
void sgemm_naive(
    size_t M, size_t N, size_t K,
    const float& alpha,
    const float* A, size_t lda, bool trana,
    const float* B, size_t ldb, bool tranb,
    const float& beta,
    float* C, size_t ldc
) {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

    kernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, lda, trana, B, ldb, tranb, beta, C, ldc);
}

CLANG_IGNORE_IGNORED_ATTRIBUTES_PUSH
#define instantiate(BLOCK_SIZE) template AXM_API void sgemm_naive<BLOCK_SIZE>(size_t,size_t,size_t,const float&,const float*,size_t,bool,const float*,size_t,bool,const float&,float*,size_t);
instantiate(16)
instantiate(32)
CLANG_IGNORE_IGNORED_ATTRIBUTES_POP

}