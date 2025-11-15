#pragma once

#include "AXM/Tensor.hpp"

namespace axm::tensor {

template<typename T>
Tensor<T>& fill(Tensor<T>& tensor, T value);

template<typename T>
Tensor<T> full(std::initializer_list<size_t> dims, T value, Device device = CPU);
template<typename T>
Tensor<T> zeros(std::initializer_list<size_t> dims, Device device = CPU);
template<typename T>
Tensor<T> ones(std::initializer_list<size_t> dims, Device device = CPU);

template<typename T>
Tensor<T> range(size_t end, Device device = CPU);
template<typename T>
Tensor<T> range(T start, T end, T step = 1, Device device = CPU);

}