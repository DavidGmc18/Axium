#include "AXM/types.hpp"

namespace axm {

std::ostream& operator<<(std::ostream& os, const half& h) {
    os << __half2float(h);
    return os;
}

std::ostream& operator<<(std::ostream& os, const __nv_bfloat16& bf) {
    os << __bfloat162float(bf);
    return os;
}

}