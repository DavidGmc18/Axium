#pragma once

#include "api.hpp"
#include <cstdlib>
#include <initializer_list>
#include <ostream>

namespace axm {

struct AXM_API TensorDescriptor {
protected:
    bool managed_desc;
    void set_descriptor(const TensorDescriptor& desc);
public:
    size_t ndim;
    size_t* dims;
    size_t* strides;
    size_t size;

    TensorDescriptor();
    TensorDescriptor(size_t ndim_);
    TensorDescriptor(std::initializer_list<size_t> dims_);
    TensorDescriptor(const TensorDescriptor& desc);
    TensorDescriptor(const TensorDescriptor* desc);
    ~TensorDescriptor();

    bool operator==(const TensorDescriptor& desc) const;
    bool operator!=(const TensorDescriptor& desc) const;

    friend AXM_API std::ostream& operator<<(std::ostream& os, const TensorDescriptor& t);
};

}