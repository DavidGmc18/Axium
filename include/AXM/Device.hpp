#pragma once

#include "api.hpp"
#include <ostream>

namespace axm {

enum Device {
    CPU  = 0,
    CUDA = 1
};

AXM_API std::ostream& operator<<(std::ostream& os, Device device);

}