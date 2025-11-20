#pragma once

#include "AXM/api.hpp"

namespace axm::op {
    AXM_API void sgemm_naive(
        size_t M, size_t N, size_t K,
        const float& alpha,
        const float* A, size_t lda, bool trana,
        const float* B, size_t ldb, bool tranb,
        const float& beta,
        float* C, size_t ldc
    );

    template<const size_t CM, const size_t CN, const size_t CK>
    AXM_API void sgemm(
        size_t M, size_t N, size_t K,
        const float& alpha,
        const float* A, size_t lda, bool trana,
        const float* B, size_t ldb, bool tranb,
        const float& beta,
        float* C, size_t ldc
    );
    
    namespace cuda {
        template<const int BLOCK_SIZE>
        AXM_API void sgemm_naive(
            size_t M, size_t N, size_t K,
            const float& alpha,
            const float* A, size_t lda, bool trana,
            const float* B, size_t ldb, bool tranb,
            const float& beta,
            float* C, size_t ldc
        );
    }
}