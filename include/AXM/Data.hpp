#pragma once

#include "api.hpp"
#include <stddef.h>

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

AXM_API size_t size_of_dtype(Dtype dtype);

AXM_API const char* dtype_to_string(Dtype dtype);

}