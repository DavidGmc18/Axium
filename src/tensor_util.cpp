// #include "AXM/tensor_util.hpp"
// #include "util.hpp"

// namespace axm::tensor {
//     template<typename T>
//     Tensor full(T value, std::initializer_list<size_t> dims, Dtype dtype, Device device) {
//         // Assert numeric
//         Tensor tensor(dims, dtype, device);

//         switch (device) {
//             case CPU: 
//                 memset(tensor.data, value, tensor.bytes);
//                 break;
//             case CUDA: 
//                 CUER(cudaMemset(tensor.data, value, tensor.bytes));
//                 break;
//             default: 
//                 std::cerr << "Unknown device\n";
//                 ABORT_DEBUG()
//         }

//         return tensor;
//     }
// }