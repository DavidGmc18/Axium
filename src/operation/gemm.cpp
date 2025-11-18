#include "AXM/operations.hpp"
#include <immintrin.h>
#include <cstring>
#include <cstdio>
#include <algorithm>

#define CM 192
#define CN 192
#define CK 192

namespace axm::op {

__forceinline void load_At(float tile[CM][CK], const float* A, size_t lda, size_t m, size_t k, size_t M, size_t K) {
    size_t col = std::min(size_t(CK), K - k);
    size_t rows = std::min(size_t(CM), M - m);
    for (size_t mm = 0; mm < rows; ++mm) {
        for (size_t kk = 0; kk < col; ++kk) {
            tile[mm][kk] = A[(m + mm)*lda + k + kk];
        }
    }
} 

__forceinline void load_Bt(float tile[CK][CN], const float* B, size_t ldb, size_t k, size_t n, size_t K, size_t N) {
    size_t col = std::min(size_t(CN), N - n);
    size_t rows = std::min(size_t(CK), K - k);
    for (size_t kk = 0; kk < rows; ++kk) {
        for (size_t nn = 0; nn < col; ++nn) {
            tile[kk][nn] = B[(k + kk)*ldb + n + nn];
        }
    }
}

__forceinline void kernel_6x16(size_t mm, size_t nn, size_t kk, const float At[CM][CK], const float Bt[CK][CN], __m256 (&acc)[6][2]) {
    __m256 b0 = _mm256_load_ps(&Bt[kk][nn]);
    __m256 b1 = _mm256_load_ps(&Bt[kk][nn + 8]);

    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        __m256 a = _mm256_set1_ps(At[mm + i][kk]);
        acc[i][0] = _mm256_fmadd_ps(a, b0, acc[i][0]);
        acc[i][1] = _mm256_fmadd_ps(a, b1, acc[i][1]);
    }
}

__forceinline void store_6x16(__m256 (&acc)[6][2], float* C, size_t ldc, const float alpha, const float beta, size_t m, size_t n, size_t M, size_t N) {
    __m256 a = _mm256_set1_ps(alpha);
    __m256 b = _mm256_set1_ps(beta);

    uint32_t cols = N - n;
    __m256i mask[2] = {_mm256_setr_epi32(0,1,2,3,4,5,6,7), _mm256_setr_epi32(8,9,10,11,12,13,14,15)};
    mask[0] = _mm256_cmpgt_epi32(_mm256_set1_epi32(cols), mask[0]);
    mask[1] = _mm256_cmpgt_epi32(_mm256_set1_epi32(cols), mask[1]);

    const size_t rows = std::min(size_t(6), M - m);
    for (size_t mm = 0; mm < rows; ++mm) {
        if (m + mm >= M) return;
        acc[mm][0] = _mm256_mul_ps(acc[mm][0], a);
        acc[mm][1] = _mm256_mul_ps(acc[mm][1], a);

        __m256 c[] = {
            _mm256_load_ps(C + (m + mm) * ldc + n),
            _mm256_load_ps(C + (m + mm) * ldc + n + 8)
        };

        acc[mm][0] = _mm256_fmadd_ps(b, c[0], acc[mm][0]);
        acc[mm][1] = _mm256_fmadd_ps(b, c[1], acc[mm][1]);
        
        _mm256_maskstore_ps(C + (m + mm) * ldc + n, mask[0], acc[mm][0]);
        _mm256_maskstore_ps(C + (m + mm) * ldc + n + 8, mask[1], acc[mm][1]);
    }
}

__forceinline void kernel_tile(
    size_t M, size_t N, size_t K,
    size_t m, size_t n, size_t k,
    const float& alpha,
    const float* A, size_t lda, bool trana,
    const float* B, size_t ldb, bool tranb,
    const float& beta,
    float* C, size_t ldc
) {
    // TODO needs zeroing out with not nice K dim, optimize
    alignas(32) float At[CM][CK] = {};
    alignas(32) float Bt[CK][CN] = {};

    load_At(At, A, lda, m, k, M, K);
    load_Bt(Bt, B, ldb, k, n, K, N);

    #pragma unroll
    for (size_t mm = 0; mm < CM; mm += 6) {
        #pragma unroll
        for (size_t nn = 0; nn < CN; nn += 16) {
            __m256 acc[6][2] = {};
            for (size_t kk = 0; kk < CK; ++kk) {
                kernel_6x16(mm, nn, kk, At, Bt, acc);
            }
            store_6x16(acc, C, ldc, alpha, (k==0 ? beta : 1.0f), m + mm, n + nn, M, N);
        }  
    }
}

// TODO no beta so far
// TOOD no transpose
void sgemm(
    size_t M, size_t N, size_t K,
    const float& alpha,
    const float* A, size_t lda, bool trana,
    const float* B, size_t ldb, bool tranb,
    const float& beta,
    float* C, size_t ldc
) {
    for (size_t m = 0; m < M; m += CM) {
        for (size_t n = 0; n < N; n += CN) {
            for (size_t k = 0; k < K; k += CK) {
                kernel_tile(
                    M, N, K,
                    m, n, k,
                    alpha,
                    A, lda, trana,
                    B, ldb, tranb,
                    beta,
                    C, ldc
                );
            }
        }
    } 

    // for (size_t mn = 0; mn < (M/CM)*(N/CN); ++mn) {
    //     size_t m = mn / (N/CN) * CM;
    //     size_t n = mn % (N/CN) * CN;
    //     for (size_t k = 0; k < K; k += CK) {
    //         kernel_tile(
    //             M, N, K,
    //             m, n, k,
    //             alpha,
    //             A, lda, trana,
    //             B, ldb, tranb,
    //             beta,
    //             C, ldc
    //         );
    //     }
    // } 
}

}