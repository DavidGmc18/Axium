#pragma once

#include <ostream>

namespace axm {

enum Device {
    CPU  = 0,
    CUDA = 1
};

std::ostream& operator<<(std::ostream& os, Device device);

}