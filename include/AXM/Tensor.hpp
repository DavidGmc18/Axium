#pragma once

#include "Device.hpp"
#include "TensorDescriptor.hpp"

namespace axm {
    
struct AXM_API Tensor : TensorDescriptor {
protected:
    bool managed;
    void alloc();
    void dealloc();
    void copy_data(void* src);
public:
    void* data;
    Device device;

    Tensor();
    Tensor(std::initializer_list<size_t> dims_, Dtype dtype_ = FP32, Device device_ = CPU);
    Tensor(TensorDescriptor& desc, Device device_ = CPU);
    Tensor(Tensor& tensor, bool copy = true);

    ~Tensor();

    void toCuda();
    void fromCuda();

    const TensorDescriptor& descriptor() const;
   
    const bool get_managed() const;

    Tensor& operator=(const Tensor& tensor);
    Tensor& operator=(const Tensor* tensor);
    
    template<typename T>
    Tensor& operator=(const std::initializer_list<T> list);

    friend AXM_API std::ostream& operator<<(std::ostream& os, const Tensor& t);
};

}