#pragma once

#include "Data.hpp"
#include <cstdlib>
#include <initializer_list>
#include <ostream>

namespace axm {

struct TensorDescriptor {
    size_t ndim;
    size_t* dims;
    size_t* strides;
    Dtype dtype;
    size_t size;
    size_t bytes;

    TensorDescriptor(std::initializer_list<size_t> dims_, Dtype dtype_, size_t alignment);
    ~TensorDescriptor();

    friend std::ostream& operator<<(std::ostream& os, const TensorDescriptor& t);
};

}