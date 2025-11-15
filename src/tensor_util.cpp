#include "AXM/tensor_util.hpp"
#include "AXM/types.hpp"
#include "util.hpp"
#include "util.hpp"
#include <iostream>

namespace axm::tensor {

template<typename T>
Tensor<T>& fill(Tensor<T>& tensor, T value) {
    switch(tensor.device) {
        case CPU: 
            std::fill(tensor.data, tensor.data + tensor.size, value);
            break;
        // TODO
        // case CUDA:
        //     std::fill(tensor.data, tensor.data + tensor.size, value); 
        //     CUER(cudaMemset(tensor.data, value, tensor.size * sizeof(T)));
        //     break;
        default:
            std::cerr << "Unknown device\n";
            ABORT_DEBUG()
    }
    return tensor;
}

template<typename T>
Tensor<T> full(std::initializer_list<size_t> dims, T value, Device device) {
    Tensor<T> tensor(dims, device);
    fill(tensor, value);
    return tensor;
}

template<typename T>
Tensor<T> zeros(std::initializer_list<size_t> dims, Device device) {
    return full<T>(dims, 0, device);
}

template<typename T>
Tensor<T> ones(std::initializer_list<size_t> dims, Device device) {
    return full<T>(dims, 1, device);
}

template<typename T>
Tensor<T> range(size_t end, Device device) {
    if (end <= 0) throw std::invalid_argument("end must be positive");
    Tensor<T> tensor({end}, device);
    for (size_t i = 0; i < tensor.size; ++i) {
        tensor.data[i] = i;
    }
    return tensor;
}

template<typename T>
Tensor<T> range(T start, T end, T step, Device device) {
    if (step == T(0)) throw std::invalid_argument("step cannot be zero");
    size_t len = std::ceil(double(end - start) / double(step));
    if (len <= 0) throw std::invalid_argument("invalid range: start, end, and step combination results in empty tensor");
    Tensor<T> tensor({len}, device);
    for (size_t i = 0; i < tensor.size; ++i) {
        tensor.data[i] = start + T(i) * step;
    }
    return tensor;
}

#define X(DTYPE, T) \
    template AXM_API Tensor<T>& axm::tensor::fill(Tensor<T>&, T); \
    template AXM_API Tensor<T> axm::tensor::full(std::initializer_list<size_t>, T, Device); \
    template AXM_API Tensor<T> axm::tensor::zeros(std::initializer_list<size_t>, Device); \
    template AXM_API Tensor<T> axm::tensor::ones(std::initializer_list<size_t>, Device); \
    template AXM_API Tensor<T> axm::tensor::range(size_t, Device); \
    template AXM_API Tensor<T> axm::tensor::range(T, T, T, Device);

AXM_TYPE_LIST
#undef X

}