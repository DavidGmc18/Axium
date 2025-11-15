#include "AXM/Tensor.hpp"
#include "AXM/Data.hpp"
#include "util.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cuda_runtime.h>
#include <numeric>
#include <sstream>

#ifdef _MSC_VER
    #include <malloc.h>
    #define ALIGNED_ALLOC(align, size) _aligned_malloc(size, align)
    #define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
    #define ALIGNED_ALLOC(align, size) aligned_alloc(align, size)
    #define ALIGNED_FREE(ptr) free(ptr)
#endif

#define CUER(call)                                                                                 \
do {                                                                                               \
    cudaError_t err = call;                                                                        \
    if (err != cudaSuccess) {                                                                      \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                                                        \
    }                                                                                              \
} while(0)

namespace axm {

void Tensor::alloc() {
    switch (device) {
        case CPU: 
            data = ALIGNED_ALLOC(ALIGNMENT, bytes); 
            break;
        case CUDA: 
            CUER(cudaMalloc((void**)&data, bytes));
            break;
        default: 
            std::cerr << "Unknown device\n";
            ABORT_DEBUG()
    }
    PTR_CHECK(data)
    managed = true;
}

void Tensor::dealloc() {
    if (managed) {
        switch (device) {
            case CPU: ALIGNED_FREE(data); break;
            case CUDA: CUER(cudaFree(data)); break;
            default:
                std::cerr << "Unknown device, can't dealloc tensor\n";
                ABORT_DEBUG()
        }
    }
    managed = false;
}

void Tensor::copy_data(void* src) {
    switch (device) {
        case CPU: 
            memcpy(data, src, bytes);
            break;
        case CUDA: 
            CUER(cudaMemcpy(data, src, bytes, cudaMemcpyDeviceToDevice));
            break;
        default:
            std::cerr << "Unknown device\n";
            ABORT_DEBUG()
    }
}

Tensor::Tensor(): TensorDescriptor(), managed(false) {
}

Tensor::Tensor(std::initializer_list<size_t> dims_, Dtype dtype_, Device device_)
: TensorDescriptor(dims_, dtype_), device(device_), managed(false) {
    alloc();
}

Tensor::Tensor(TensorDescriptor& desc, Device device_):
TensorDescriptor(desc), managed(false) {
    alloc();
}

Tensor::Tensor(Tensor& tensor, bool copy): TensorDescriptor(tensor.descriptor()), device(tensor.device), managed(false) {
    if (copy) {
        alloc();
        copy_data(tensor.data);
    } else {
        data = tensor.data;
    }
}

Tensor::~Tensor() {
    dealloc();
}

void Tensor::toCuda() {
    void* new_data;

    CUER(cudaMalloc((void**)&new_data, bytes));
    CUER(cudaMemcpy(new_data, data, bytes, cudaMemcpyHostToDevice));
    ALIGNED_FREE(data);

    data = new_data;
    device = CUDA;
}

void Tensor::fromCuda() {
    void* new_data = ALIGNED_ALLOC(32, bytes);
    CUER(cudaMemcpy(new_data, data, bytes, cudaMemcpyDeviceToHost));
    CUER(cudaFree(data));

    data = new_data;
    device = CPU;
}

const TensorDescriptor& Tensor::descriptor() const {
    return *this;
}

const bool Tensor::get_managed() const {
    return managed;
}

Tensor& Tensor::operator=(const Tensor& tensor) {
    if (this->descriptor() != tensor.descriptor()) {
        dealloc();
        set_descriptor(tensor);
        device = tensor.device;
        alloc();
    }
    if (this->data == tensor.data) return *this;
    copy_data(tensor.data);
    return *this;
}

Tensor& Tensor::operator=(const Tensor* tensor) {
    if (this->descriptor() != tensor->descriptor()) {
        std::cerr << "Assignment failed for shallow copy: the descriptors of source and destination tensors do not match\n";
        ABORT_DEBUG()
        return *this;
    }
    data = tensor->data;
    return *this;
}

// TODO assumes padding only on last dim and defualt ordered dims/strides
template<typename T>
Tensor& Tensor::operator=(const std::initializer_list<T> list) {
    size_t list_bytes = list.size() * sizeof(T);
    if (list_bytes > bytes) list_bytes = bytes;

    size_t inner = dims[ndim - 1] * size_of_dtype(dtype);
    size_t outer = list.size() / dims[ndim - 1];
    size_t stride = strides[ndim - 2] * size_of_dtype(dtype);
    size_t remainder = list_bytes % inner;

    const char* src = reinterpret_cast<const char*>(list.begin());
    char* dst = (char*)data;

    switch (dtype) {
        #define X(DTYPE, TYPE) case DTYPE:\
            if constexpr (std::is_same_v<TYPE, T>) {\
                for (size_t i = 0; i < outer; ++i) {\
                    memcpy(dst, src, inner);\
                    dst += stride;\
                    src += inner;\
                }\
                if (remainder) memcpy(dst, src, remainder);\
                return *this;\
            }\
            break;
        AXM_DTYPE_LIST
        #undef X
        default: break;
    }

    inner /= size_of_dtype(dtype);
    remainder /= size_of_dtype(dtype);
    size_t stride_src = inner * sizeof(T);
    switch (dtype) {
        #define X(DTYPE, TYPE) case DTYPE:\
            if constexpr (std::is_convertible_v<T, TYPE>) {\
                for (size_t i = 0; i < outer; ++i) {\
                    for (size_t j = 0; j < inner; ++j) {\
                        reinterpret_cast<TYPE*>(dst)[j] = static_cast<TYPE>(src[j * sizeof(T)]);\
                    }\
                    dst += stride;\
                    src += stride_src;\
                }\
                if (remainder)\
                    for (size_t j = 0; j < remainder; ++j) {\
                        reinterpret_cast<TYPE*>(dst)[j] = static_cast<TYPE>(src[j * sizeof(T)]);\
                    }\
                return *this;\
            }
        AXM_DTYPE_LIST
        #undef X
        default: break;
    }

    std::cerr << "Cannot assign initializer_list of type '" + std::string(typeid(T).name()) + "' to Tensor with dtype '" + dtype_to_string(dtype) + "'";
    ABORT_DEBUG()

    return *this;
}
#define X(DTYPE, TYPE) template AXM_API Tensor& Tensor::operator=<TYPE>(std::initializer_list<TYPE>);
AXM_DTYPE_LIST
#undef X

std::ostream& operator<<(std::ostream& os, const axm::Tensor& tensor) {
    std::ostringstream buffer;
    size_t* ids = new size_t[tensor.ndim];
    memset(ids, 0, tensor.ndim * sizeof(size_t));

    size_t depth = 0;
    for (size_t i = 0; i < tensor.size; ++i) {
        for (size_t dim = tensor.ndim; dim-- > 0;) {
            if (depth > 0 && depth < tensor.ndim) buffer << '\n';
            if (ids[dim] != 0) break;
            buffer << std::string(depth, ' ') << '[';
            depth++;
        }

        size_t idx = std::inner_product(ids, ids + tensor.ndim, tensor.strides, size_t(0));
        switch (tensor.dtype) {
            #define X(DTYPE, TYPE) case DTYPE: buffer << reinterpret_cast<TYPE*>(tensor.data)[idx]; break;
            AXM_DTYPE_LIST
            #undef X
            default: buffer << "UNK"; std::cerr << "Unknown dtype\n";
        }

        for (size_t dim = tensor.ndim; dim-- > 0;) {
            if (dim + 1 != tensor.ndim) {
                if (depth < tensor.ndim) buffer << '\n';
                depth--;
                if (depth + 1 == tensor.ndim) buffer << ']';
                else buffer << std::string(depth, ' ') << ']';   
            }
            if (ids[dim] + 1 < tensor.dims[dim]) buffer << ", ";

            ids[dim]++;
            if (ids[dim] < tensor.dims[dim]) break;
            ids[dim] = 0;
        }
    }
    
    free(ids);

    if (tensor.ndim > 1) buffer << "\n]"; 
    else if (tensor.ndim == 1) buffer << ']';

    return os << buffer.str();
}

}