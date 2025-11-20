#include "AXM/Tensor.hpp"
#include "AXM/operations.hpp"
#include <chrono>
#include <random>

int main() {
    const size_t S = 960;
    const size_t M = S, N = S, K = S;
    axm::Tensor<float> a({M, K});
    axm::Tensor<float> b({K, N});
    axm::Tensor<float> c({M, N}, axm::CUDA);
    axm::Tensor<float> d({M, N});

    float alpha = 1.0f;
    float beta = 0.0f;

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0.0f, 100.0f);

    for (size_t i = 0; i < a.size; ++i) {
        a.data[i] = dist(rng);
    }
    for (size_t i = 0; i < b.size; ++i) {
        b.data[i] = dist(rng);
    }

    axm::op::sgemm_naive(
        M, N, K,
        alpha,
        a.data, a.strides[0], false,
        b.data, b.strides[0], false,
        beta,
        d.data, d.strides[0]
    );

    a.toCuda();
    b.toCuda();

    auto start = std::chrono::high_resolution_clock::now();
    axm::op::cuda::sgemm_naive<32>(
        M, N, K,
        alpha,
        a.data, a.strides[0], false,
        b.data, b.strides[0], false,
        beta,
        c.data, c.strides[0]
    );
    auto end = std::chrono::high_resolution_clock::now();

    c.fromCuda();

    double avg = 0;
    double max = 0;
    for (size_t i = 0; i < d.size; ++i) {
        double error = std::abs(double(d.data[i] - c.data[i]) / double(d.data[i]));
        avg += error;
        max = std::max(max, error);
    }
    avg /= d.size;
    printf("ERROR -> AVG: %f%%    MAX: %f%%\n", avg*100, max*100);

    std::chrono::duration<double> duration = end - start;
    double elapsed = duration.count();
    double num_flops = 2 * N * N * N;
    double flops = num_flops / elapsed;
    // double relative_performance = flops / (4.4e9 * 32); // Ryzen 7 5800H
    double relative_performance = flops / (15.97e12); // RTX 3070 8GB mobile
    printf("GFLOP: %f    Elapsed: %fms    GFLOP/S: %f    Relative: %f%%", num_flops / 1e9, elapsed * 1e3, flops / 1e9, relative_performance * 100);
}