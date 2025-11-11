#include "AXM/Tensor.hpp"
#include "AXM/Data.hpp"
#include "util.hpp"
#include <cstdint>
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

        size_t idx = std::inner_product(ids, ids + tensor.ndim, tensor.strides, 0);
        switch (tensor.dtype) {
            case FP32:   buffer << static_cast<float*>(tensor.data)[idx]; break;
            case FP64:   buffer << static_cast<double*>(tensor.data)[idx]; break;
            case FP16:   buffer << "UNK"; break;
            case BF16:   buffer << "UNK"; break;
            case BOOL:   buffer << static_cast<bool*>(tensor.data)[idx]; break;
            case INT8:   buffer << static_cast<int8_t*>(tensor.data)[idx]; break;
            case INT32:  buffer << static_cast<int32_t*>(tensor.data)[idx]; break;
            case UINT8:  buffer << static_cast<uint8_t*>(tensor.data)[idx]; break;
            case UINT32: buffer << static_cast<uint32_t*>(tensor.data)[idx]; break;
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