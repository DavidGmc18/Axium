#pragma once

#include "api.hpp"

namespace axm::op {
    AXM_API void sgemm_naive(
        size_t M, size_t N, size_t K,
        float& alpha,
        const float* A, size_t lda, bool trana,
        const float* B, size_t ldb, bool tranb,
        float& beta,
        float* C, size_t ldc
    );

    AXM_API void sgemm(
        size_t M, size_t N, size_t K,
        float& alpha,
        const float* A, size_t lda, bool trana,
        const float* B, size_t ldb, bool tranb,
        float& beta,
        float* C, size_t ldc
    );
}