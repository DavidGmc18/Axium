#include "AXM/TensorDescriptor.hpp"
#include "util.hpp"
#include <sstream>
#include <iostream>

inline constexpr size_t ceil(size_t value, size_t step) {
    return ((value + step - 1) / step) * step;
}

namespace axm {

TensorDescriptor::TensorDescriptor(): ndim(0), size(0), managed_desc(false) {
}

TensorDescriptor::TensorDescriptor(size_t ndim_): ndim(ndim_), managed_desc(true) {
    dims = (size_t*)malloc(sizeof(size_t) * ndim);
    strides = (size_t*)malloc(sizeof(size_t) * ndim);
}

TensorDescriptor::TensorDescriptor(std::initializer_list<size_t> dims_): TensorDescriptor(dims_.size()) {
    size = ndim ? 1 : 0;

    size_t idx = 0;
    for (size_t d : dims_) dims[idx++] = d;

    for (size_t i = ndim; i-- > 0;) {
        strides[i] = size;
        size *= dims[i];
    }
}

TensorDescriptor::TensorDescriptor(const TensorDescriptor& desc): TensorDescriptor(desc.ndim) {
    size = desc.size;

    memcpy(dims, desc.dims, sizeof(size_t) * ndim);
    memcpy(strides, desc.strides, sizeof(size_t) * ndim);
}

TensorDescriptor::TensorDescriptor(const TensorDescriptor* desc):
ndim(desc->ndim), size(desc->size), managed_desc(false) {
    dims = desc->dims;
    strides = desc->strides;
}

TensorDescriptor::~TensorDescriptor() {
    if (managed_desc && ndim > 0) {
        free(dims);
        dims = nullptr;
        free(strides);
        strides = nullptr;
    }
}

void TensorDescriptor::set_descriptor(const TensorDescriptor& desc) {
    if (*this == desc) return;

    ndim = desc.ndim;
    size = desc.size;;

    if (!managed_desc) {
        dims = (size_t*)malloc(sizeof(size_t) * ndim);
        strides = (size_t*)malloc(sizeof(size_t) * ndim);
        managed_desc = true;
    } else {
        dims = (size_t*)realloc(dims, sizeof(size_t) * ndim);
        strides = (size_t*)realloc(strides, sizeof(size_t) * ndim);
    }
    PTR_CHECK(dims)
    PTR_CHECK(strides)
    memcpy(dims, desc.dims, sizeof(size_t) * ndim);
    memcpy(strides, desc.strides, sizeof(size_t) * ndim);
}

bool TensorDescriptor::operator==(const TensorDescriptor& desc) const {
    if (ndim != desc.ndim || size != desc.size) return false;
    return std::equal(dims, dims + ndim, desc.dims) && std::equal(strides, strides + ndim, desc.strides);
}

bool TensorDescriptor::operator!=(const TensorDescriptor& desc) const {
    return !(*this == desc);
}

std::ostream& operator<<(std::ostream& os, const axm::TensorDescriptor& descriptor) {
    std::ostringstream buffer;
    buffer << "TensorDescriptor(";
    buffer << "ndim=" << descriptor.ndim << ", ";

    buffer << "dims={";
    for (size_t i = 0; i < descriptor.ndim; i++) {
        if (i > 0) buffer << ", ";
        buffer << descriptor.dims[i];
    }
    buffer << "}, ";

    buffer << "strides={";
    for (size_t i = 0; i < descriptor.ndim; i++) {
        if (i > 0) buffer << ", ";
        buffer << descriptor.strides[i];
    }
    buffer << "}, ";

    buffer << "size=" << descriptor.size;
    buffer << ")";
    return os << buffer.str();
}

}