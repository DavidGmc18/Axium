#include "AXM/Data.hpp"

#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

namespace axm {

size_t size_of_dtype(Dtype dtype) {
    switch (dtype) {
        #define X(DTYPE, TYPE) case DTYPE: return sizeof(TYPE);
                AXM_DTYPE_LIST
        #undef X
        default:
            fprintf(stderr, "Unknown Dtype %d\n", dtype);
            return 0;
    }
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

std::ostream& operator<<(std::ostream& os, const half& h) {
    os << __half2float(h);
    return os;
}

std::ostream& operator<<(std::ostream& os, const __nv_bfloat16& bf) {
    os << __bfloat162float(bf);
    return os;
}

}