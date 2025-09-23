#include "AXM/Device.hpp"

std::ostream& axm::operator<<(std::ostream& os, Device device) {
    switch(device) {
        case CPU:  return os << "CPU";
        case CUDA: return os << "CUDA";
        default:   return os << "UNKNOWN";
    }
}