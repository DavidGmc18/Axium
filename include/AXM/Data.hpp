#pragma once

#include "api.hpp"
#include <stddef.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>

#define AXM_DTYPE_LIST \
    X(FP32, float)\
    X(FP64, double)\
    X(FP16, half)\
    X(BF16, __nv_bfloat16)\
    X(BOOL, bool)\
    X(INT8, int8_t)\
    X(INT32, int32_t)\
    X(UINT8, uint8_t)\
    X(UINT32, uint32_t)

namespace axm {
    
enum Dtype {
    NONE   = -1,
    FP32   = 0,
    FP64   = 1,
    FP16   = 2,
    BF16   = 3,
    BOOL   = 4,
    INT8   = 5,
    INT32  = 6,
    UINT8  = 7,
    UINT32 = 8
};

template<axm::Dtype DT> struct dtype_to_type;

AXM_API size_t size_of_dtype(Dtype dtype);
AXM_API const char* dtype_to_string(Dtype dtype);

std::ostream& operator<<(std::ostream& os, const half& h);
std::ostream& operator<<(std::ostream& os, const __nv_bfloat16& bf);

}