#include "AXM/operations.hpp"
#include <immintrin.h>
#include <cstring>
#include <cstdio>

#define CM 192
#define CN 192
#define CK 192

namespace axm::op {

void sgemm_naive(
    size_t M, size_t N, size_t K,
    float& alpha,
    const float* A, size_t lda, bool trana,
    const float* B, size_t ldb, bool tranb,
    float& beta,
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

__forceinline void load_At(float tile[CM][CK], const float* A, size_t lda, size_t m, size_t k, size_t M, size_t K) {
    if (CM <= (M - m)) {
        if (CK <= (K - k)) {
            #pragma unroll
            for (size_t mm = 0; mm < CM; ++mm) {
                memcpy(tile[mm], A + (m + mm) * lda + k, CK * sizeof(float));
            }
        } else {
            size_t bytes = (K - k) * sizeof(float);
            #pragma unroll
            for (size_t mm = 0; mm < CM; ++mm) {
                memcpy(tile[mm], A + (m + mm) * lda + k, bytes);
            }
        }
    } else {
        if (CK <= (K - k)) {
            for (size_t mm = 0; mm < CM; ++mm) {
                memcpy(tile[mm], A + (m + mm) * lda + k, CK * sizeof(float));
            }
        } else {
            size_t bytes = (K - k) * sizeof(float);
            for (size_t mm = 0; mm < CM; ++mm) {
                memcpy(tile[mm], A + (m + mm) * lda + k, bytes);
            }
        }
    }
} 

__forceinline void load_Bt(float tile[CK][CN], const float* B, size_t ldb, size_t k, size_t n, size_t K, size_t N) {
    if (CK <= (K - k)) {
        if (CN <= (N - n)) {
            #pragma unroll
            for (size_t kk = 0; kk < CK; ++kk) {
                memcpy(tile[kk], B + (k + kk) * ldb + n, CN * sizeof(float));
            }
        } else {
            size_t bytes = (N - n) * sizeof(float);
            #pragma unroll
            for (size_t kk = 0; kk < CK; ++kk) {
                memcpy(tile[kk], B + (k + kk) * ldb + n, bytes);
            }
        }
    } else {
        if (CN <= (N - n)) {
            for (size_t kk = 0; kk < CK; ++kk) {
                memcpy(tile[kk], B + (k + kk) * ldb + n, CN * sizeof(float));
            }
        } else {
            size_t bytes = (N - n) * sizeof(float);
            for (size_t kk = 0; kk < CK; ++kk) {
                memcpy(tile[kk], B + (k + kk) * ldb + n, bytes);
            }
        }
    }
}

__forceinline void kernel_6x16(size_t m, size_t n, size_t k, const float At[CM][CK], const float Bt[CK][CN], __m256 (&acc)[6][2]) {
    __m256 b0 = _mm256_load_ps(&Bt[k][n]);
    __m256 b1 = _mm256_load_ps(&Bt[k][n + 8]);

    #pragma unroll
    for (int mm = 0; mm < 6; ++mm) {
        __m256 a = _mm256_set1_ps(At[m + mm][k]);
        acc[mm][0] = _mm256_fmadd_ps(a, b0, acc[mm][0]);
        acc[mm][1] = _mm256_fmadd_ps(a, b1, acc[mm][1]);
    }
}

__forceinline void store_6x16(__m256 (&acc)[6][2], float* C, size_t ldc, float alpha, float beta, size_t m, size_t n) {
    __m256 a = _mm256_set1_ps(alpha);
    __m256 b = _mm256_set1_ps(beta);

    #pragma unroll
    for (size_t mm = 0; mm < 6; ++mm) {
        acc[mm][0] = _mm256_mul_ps(acc[mm][0], a);
        acc[mm][1] = _mm256_mul_ps(acc[mm][1], a);

        __m256 c[] = {
            _mm256_load_ps(C + (m + mm) * ldc + n),
            _mm256_load_ps(C + (m + mm) * ldc + n + 8)
        };

        acc[mm][0] = _mm256_fmadd_ps(b, c[0], acc[mm][0]);
        acc[mm][1] = _mm256_fmadd_ps(b, c[1], acc[mm][1]);

        _mm256_store_ps(C + (m + mm) * ldc + n, acc[mm][0]);
        _mm256_store_ps(C + (m + mm) * ldc + n + 8, acc[mm][1]);
    }
}

__forceinline void kernel_tile(
    size_t M, size_t N, size_t K,
    size_t m, size_t n, size_t k,
    float& alpha,
    const float* A, size_t lda, bool trana,
    const float* B, size_t ldb, bool tranb,
    float& beta,
    float* C, size_t ldc
) {
    alignas(32) float At[CM][CK];
    alignas(32) float Bt[CK][CN];

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
            store_6x16(acc, C, ldc, alpha, (k==0 ? beta : 1.0f), m + mm, n + nn);
        }  
    }
}

// TODO no beta so far
// TOOD no transpose
void sgemm(
    size_t M, size_t N, size_t K,
    float& alpha,
    const float* A, size_t lda, bool trana,
    const float* B, size_t ldb, bool tranb,
    float& beta,
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
}

}