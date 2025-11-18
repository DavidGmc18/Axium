#include "AXM/operations.hpp"

namespace axm::op {

void sgemm_naive(
    size_t M, size_t N, size_t K,
    const float& alpha,
    const float* A, size_t lda, bool trana,
    const float* B, size_t ldb, bool tranb,
    const float& beta,
    float* C, size_t ldc
) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                float a = trana ? A[k * lda + m] : A[m * lda + k];
                float b = tranb ? B[n * ldb + k] : B[k * ldb + n];
                acc += a * b;
            }
            C[m * ldc + n] = alpha * acc + beta * C[m * ldc + n];
        }  
    }
}

}