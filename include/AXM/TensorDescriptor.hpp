#pragma once

#include "api.hpp"
#include "Data.hpp"
#include <cstdlib>
#include <initializer_list>
#include <ostream>

namespace axm {

struct AXM_API TensorDescriptor {
    const static size_t ALIGNMENT = 32;
protected:
    bool managed_desc;
    void set_descriptor(const TensorDescriptor& desc);
public:
    size_t ndim;
    size_t* dims;
    size_t* strides;
    Dtype dtype;
    size_t size;
    size_t bytes;

    TensorDescriptor();
    TensorDescriptor(size_t ndim_);
    TensorDescriptor(std::initializer_list<size_t> dims_, Dtype dtype_ = FP32);
    TensorDescriptor(const TensorDescriptor& desc);
    TensorDescriptor(const TensorDescriptor* desc);
    ~TensorDescriptor();

    bool operator==(const TensorDescriptor& desc) const;
    bool operator!=(const TensorDescriptor& desc) const;

    friend AXM_API std::ostream& operator<<(std::ostream& os, const TensorDescriptor& t);
};

}