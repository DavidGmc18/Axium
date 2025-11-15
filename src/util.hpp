#pragma once

#define ABORT() {fprintf(stderr, "Abort at %s:%d\n", __FILE__, __LINE__); std::abort();}

#ifdef DEBUG
    #define ABORT_DEBUG() ABORT()
#else
    #define ABORT_DEBUG()
#endif
 
#define PTR_CHECK(ptr) if (ptr == nullptr) {std::cerr << "Nullptr!\n"; ABORT_DEBUG()}

#define CUER(call)                                                                                 \
do {                                                                                               \
    cudaError_t err = call;                                                                        \
    if (err != cudaSuccess) {                                                                      \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                                                        \
    }                                                                                              \
} while(0)
