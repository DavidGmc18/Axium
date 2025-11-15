#include "AXM/operations.hpp"
#include <immintrin.h>
#include <cstdint>

namespace axm::op {

template<>
AXM_API void add<float>(const float* A, const float* B, float* C, size_t n) {
    for (size_t i = 0; i < n; i += 8) {
        __m256 a = _mm256_load_ps(A + i);
        __m256 b = _mm256_load_ps(B + i);
        __m256 c = _mm256_add_ps(a, b);
        _mm256_store_ps(C + i, c);
    } 
}

template<>
AXM_API void add<int32_t>(const int32_t* A, const int32_t* B, int32_t* C, size_t n) {
    for (size_t i = 0; i < n; i += 8) {
        __m256i a = _mm256_load_si256((__m256i*)A + i);
        __m256i b = _mm256_load_si256((__m256i*)B + i);
        __m256i c = _mm256_add_epi32(a, b);
        _mm256_store_si256((__m256i*)C + i, c);
    } 
}

template<>
AXM_API void sub<float>(const float* A, const float* B, float* C, size_t n) {
    for (size_t i = 0; i < n; i += 8) {
        __m256 a = _mm256_load_ps(A + i);
        __m256 b = _mm256_load_ps(B + i);
        __m256 c = _mm256_sub_ps(a, b);
        _mm256_store_ps(C + i, c);
    } 
}

template<>
AXM_API void sub<int32_t>(const int32_t* A, const int32_t* B, int32_t* C, size_t n) {
    for (size_t i = 0; i < n; i += 8) {
        __m256i a = _mm256_load_si256((__m256i*)A + i);
        __m256i b = _mm256_load_si256((__m256i*)B + i);
        __m256i c = _mm256_sub_epi32(a, b);
        _mm256_store_si256((__m256i*)C + i, c);
    } 
}

}