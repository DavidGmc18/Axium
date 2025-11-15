#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>

#define AXM_TYPE_LIST \
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

std::ostream& operator<<(std::ostream& os, const half& h);
std::ostream& operator<<(std::ostream& os, const __nv_bfloat16& bf);

}