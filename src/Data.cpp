#include "AXM/Data.hpp"

#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

static const size_t DtypeSizes[9] = {
    sizeof(float),           // FLOAT32
    sizeof(double),          // FLOAT64
    sizeof(half),            // FLOAT16 (half)
    sizeof(__nv_bfloat16),   // BFLOAT16 (nv_bfloat16)
    sizeof(bool),            // BOOLEAN
    sizeof(int8_t),          // INT8
    sizeof(int32_t),         // INT32
    sizeof(uint8_t),         // UINT8
    sizeof(uint32_t)         // UINT32
};

namespace axm {

size_t size_of_dtype(Dtype dtype) {
    if (dtype < FP32 || dtype > UINT32)
        fprintf(stderr, "Unknown Dtype %d\n", dtype);
    return DtypeSizes[dtype];
}

const char* dtype_to_string(Dtype dtype) {
    switch (dtype) {
        case NONE:   return "NONE";
        case FP32:   return "FP32";
        case FP64:   return "FP64";
        case FP16:   return "FP16";
        case BF16:   return "BF16";
        case BOOL:   return "BOOL";
        case INT8:   return "INT8";
        case INT32:  return "INT32";
        case UINT8:  return "UINT8";
        case UINT32: return "UINT32";
        default: return "UNKNOWN";
    }
}

}