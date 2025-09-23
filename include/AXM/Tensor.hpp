#pragma once

#include "Device.hpp"
#include "TensorDescriptor.hpp"

namespace axm {
    
struct Tensor : TensorDescriptor {
    void* data;
    Device device;

    Tensor(std::initializer_list<size_t> dims_, Dtype dtype_ = FP32, Device device_ = CPU,
        size_t alignment_ = 32, bool zero_init = true);

    ~Tensor();

    void toCuda();
    void fromCuda();

    const TensorDescriptor& descriptor() const;

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t);
};

}