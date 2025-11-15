#include "AXM/Tensor.hpp"
#include "AXM/types.hpp"
#include "util.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cuda_runtime.h>
#include <numeric>
#include <sstream>

#define ALIGNMENT 32

#ifdef _MSC_VER
    #include <malloc.h>
    #define ALIGNED_ALLOC(size) (T*)_aligned_malloc(size, ALIGNMENT)
    #define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
    #define ALIGNED_ALLOC(size) (T*)aligned_alloc(ALIGNMENT, size)
    #define ALIGNED_FREE(ptr) free(ptr)
#endif

namespace axm {

#define X(DTYPE, TYPE) template class Tensor<TYPE>;
AXM_TYPE_LIST
#undef X

inline constexpr size_t ceil(size_t value, size_t step) {
    return ((value + step - 1) / step) * step;
}

template<typename T>
void Tensor<T>::alloc() {
    size_t bytes = ceil(size * sizeof(T), ALIGNMENT);
    switch (device) {
        case CPU: 
            data = ALIGNED_ALLOC(bytes); 
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

template<typename T>
void Tensor<T>::dealloc() {
    if (managed) {
        switch (device) {
            case CPU: ALIGNED_FREE(data); break;
            case CUDA: CUER(cudaFree(data)); break;
            default:
                std::cerr << "Unknown device, can't dealloc tensor\n";
                ABORT_DEBUG()
        }
    }
    data = nullptr;
    managed = false;
}

template<typename T>
void Tensor<T>::cpy_data(const T* src, size_t n) {
    size_t bytes = (n < size ? n : size) * sizeof(T);
    switch(device) {
        case CPU: 
            memcpy(data, src, bytes);
            break;
        case CUDA: 
            CUER(cudaMemcpy(data, src, bytes, cudaMemcpyDefault));
            break;
        default:
            std::cerr << "Unknown device\n";
            ABORT_DEBUG()
    }
}

template<typename T>
Tensor<T>::Tensor(): TensorDescriptor(), managed(false) {
}

template<typename T>
Tensor<T>::Tensor(std::initializer_list<size_t> dims_, Device device_)
: TensorDescriptor(dims_), device(device_), managed(false) {
    alloc();
}

template<typename T>
Tensor<T>::Tensor(const TensorDescriptor& desc, Device device_):
TensorDescriptor(desc), device(device_), managed(false) {
    alloc();
}

template<typename T>
Tensor<T>::Tensor(const Tensor<T>& tensor, bool copy): TensorDescriptor(tensor.descriptor()), device(tensor.device), managed(false) {
    if (copy) {
        alloc();
        cpy_data(tensor.data, size);
    } else {
        data = tensor.data;
    }
}

template<typename T>
Tensor<T>::~Tensor() {
    dealloc();
}

template<typename T>
const TensorDescriptor& Tensor<T>::descriptor() const {
    return *this;
}

template<typename T>
void Tensor<T>::toCuda() {
    T* new_data;

    CUER(cudaMalloc((void**)&new_data, size * sizeof(T)));
    CUER(cudaMemcpy(new_data, data, size * sizeof(T), cudaMemcpyHostToDevice));
    ALIGNED_FREE(data);

    data = new_data;
    device = CUDA;
}

template<typename T>
void Tensor<T>::fromCuda() {
    T* new_data = ALIGNED_ALLOC(size * sizeof(T));
    CUER(cudaMemcpy(new_data, data, size * sizeof(T), cudaMemcpyDeviceToHost));
    CUER(cudaFree(data));

    data = new_data;
    device = CPU;
}

template<typename T>
const bool Tensor<T>::get_managed() const {
    return managed;
}

template<typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& tensor) {
    if (this->descriptor() != tensor.descriptor()) {
        dealloc();
        set_descriptor(tensor);
        device = tensor.device;
        alloc();
    }
    if (this->data == tensor.data) return *this;
    cpy_data(tensor.data, tensor.size);
    return *this;
}

template<typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>* tensor) {
    if (this->descriptor() != tensor->descriptor()) {
        std::cerr << "Assignment failed for shallow copy: the descriptors of source and destination tensors do not match\n";
        ABORT_DEBUG()
        return *this;
    }
    dealloc();
    data = tensor->data;
    return *this;
}

template<typename T>
Tensor<T>& Tensor<T>::operator=(const std::initializer_list<T> list) {
    cpy_data(list.begin(), list.size());
    return *this;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor) {
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
        buffer << tensor.data[idx];

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
#define X(DTYPE, TYPE) template AXM_API std::ostream& operator<<(std::ostream&, const Tensor<TYPE>&);
AXM_TYPE_LIST
#undef X

}