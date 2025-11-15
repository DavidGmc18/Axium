#pragma once

#include "Device.hpp"
#include "TensorDescriptor.hpp"

namespace axm {

template<typename T>
struct AXM_API Tensor : TensorDescriptor {
protected:
    bool managed;
    void alloc();
    void dealloc();
    void cpy_data(const T* src, size_t n);
public:
    T* data;
    Device device;

    Tensor();
    Tensor(std::initializer_list<size_t> dims_, Device device_ = CPU);
    Tensor(TensorDescriptor& desc, Device device_ = CPU);
    Tensor(Tensor<T>& tensor, bool copy = true);

    ~Tensor();

    const TensorDescriptor& descriptor() const;

    void toCuda();
    void fromCuda();
   
    const bool get_managed() const;

    Tensor& operator=(const Tensor& tensor);
    Tensor& operator=(const Tensor* tensor);
    
    Tensor<T>& operator=(const std::initializer_list<T> list);

    template<typename U>
    friend std::ostream& operator<<(std::ostream&, const Tensor<U>&);
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& t);

}