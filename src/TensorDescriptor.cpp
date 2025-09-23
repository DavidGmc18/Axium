#include "AXM/Data.hpp"
#include "AXM/TensorDescriptor.hpp"
#include <sstream>

namespace axm {

TensorDescriptor::TensorDescriptor(std::initializer_list<size_t> dims_, Dtype dtype_, size_t alignment)
: ndim(dims_.size()), dtype(dtype_), size(1), bytes(size)
    {
        dims = (size_t*)malloc(sizeof(size_t) * ndim);
        strides = (size_t*)malloc(sizeof(size_t) * ndim);

        size_t idx = 0;
        for (auto d : dims_) dims[idx++] = d;

        for (size_t i = ndim; i-- > 0;) {
            strides[i] = bytes;
            size *= dims[i];
            if (i + 1 == ndim) {
                bytes *= (((dims[i] * size_of_dtype(dtype) + alignment - 1) / alignment) * alignment) / size_of_dtype(dtype);
            } else bytes *= dims[i];
        }

        bytes *= size_of_dtype(dtype);
    }

TensorDescriptor::~TensorDescriptor() {
    free(dims);
    free(strides);
}

std::ostream& operator<<(std::ostream& os, const axm::TensorDescriptor& descriptor) {
    std::ostringstream buffer;
    buffer << "(";

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

    buffer << "dtype=" << dtype_to_string(descriptor.dtype) << ", ";

    buffer << "size=" << descriptor.size << ", ";

    buffer << "bytes=" << descriptor.bytes;

    buffer << ")";
    return os << buffer.str();
}

}