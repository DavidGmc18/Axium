#include "AXM/Tensor.hpp"
#include "AXM/operations.hpp"
#include "AXM/tensor_util.hpp"
#include <chrono>
#include <iostream>

int main() {
    axm::Tensor<float> a = axm::tensor::full<float>({2, 2}, -1);
    axm::Tensor<float> b = axm::tensor::zeros<float>({2, 2});
    axm::Tensor<float> c = axm::tensor::ones<float>({3, 2, 1});
    axm::Tensor<float> d = axm::tensor::range<float>(1);
    axm::Tensor<float> e = axm::tensor::range<float>(10, 0, -1.9);
    axm::Tensor<float> f = axm::tensor::range<float>(10, 0, -1.99);
    axm::Tensor<float> g(f.descriptor());
    axm::op::sub<float>(e.data, f.data, g.data, g.size);

    std::cout << a << "\n\n";
    std::cout << b << "\n\n";
    std::cout << c << "\n\n";
    std::cout << d << "\n\n";
    std::cout << e << "\n\n";
    std::cout << f << "\n\n";
    std::cout << g << "\n\n";
}

// int main() {
//     const size_t N = 960;
//     axm::Tensor a({N, N});
//     axm::Tensor b({N, N});
//     axm::Tensor c({N, N});

//     c = a * b;
//     c = a * b;
//     c = a * b;

//     auto start = std::chrono::high_resolution_clock::now();
//     c = a * b;
//     auto end = std::chrono::high_resolution_clock::now();

//     std::chrono::duration<double> duration = end - start;
//     double elapsed = duration.count();
    
//     double num_flops = 2 * N * N * N;

//     double flops = num_flops / elapsed;

//     double relative_performance = flops / (4.4e9 * 32);

//     printf("GLOP: %f    Elapsed: %fms    GFLOP/S: %f    Relative: %f%%", num_flops / 1e9, elapsed * 1e3, flops / 1e9, relative_performance * 100);
// }


// int main() {
//     const size_t M = 6, N = 16, K = 2;
//     axm::Tensor<float> a({M, K});
//     axm::Tensor<float> b({K, N});

//     axm::Tensor<float> c({M, N});
//     axm::Tensor<float> d({M, N});

//     a = {
//         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
//         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
//         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
//         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
//         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
//         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
//     };

//     b = {
//         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
//         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  
//     };

//     float alpha = 1.0f;
//     float beta = 0.0f;
//     axm::op::sgemm(
//         M, N, K,
//         alpha,
//         a.data, a.strides[0], false,
//         b.data, b.strides[0], false,
//         beta,
//         c.data, c.strides[0]
//     );
//     axm::op::sgemm_naive(
//         M, N, K,
//         alpha,
//         a.data, a.strides[0], false,
//         b.data, b.strides[0], false,
//         beta,
//         d.data, d.strides[0]
//     );

//     std::cout << a << "\n\n";
//     std::cout << b << "\n\n";
//     std::cout << c << "\n\n";
//     std::cout << d << "\n\n";
// }