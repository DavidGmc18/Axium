// #include "AXM/TensorDescriptor.hpp"
#include "AXM/Tensor.hpp"
#include <iostream>

// int main() {
//     axm::TensorDescriptor b(1);
//     std::cout << b << "\n\n";

//     axm::TensorDescriptor c({1});
//     std::cout << c << "\n\n";

//     axm::TensorDescriptor d({2, 2});
//     std::cout << d << "\n\n";

//     axm::TensorDescriptor e({2, 3, 4});
//     std::cout << e << "\n\n";
// }

int main() {
    axm::Tensor<float> b({1});
    std::cout << b << "\n\n";

    axm::Tensor<float> c({2, 2});
    std::cout << c << "\n\n";

    axm::Tensor<float> d(c);
    d = {
      1, 2,
      3, 4
    };
    std::cout << d << "\n\n";

    axm::Tensor<float> e({3, 2});
    e = d;
    std::cout << e << "\n\n";
}