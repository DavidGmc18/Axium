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


#ifdef __clang__
#define CLANG_IGNORE_IGNORED_ATTRIBUTES_PUSH \
    _Pragma("clang diagnostic push") \
    _Pragma("clang diagnostic ignored \"-Wignored-attributes\"")
#define CLANG_IGNORE_IGNORED_ATTRIBUTES_POP \
    _Pragma("clang diagnostic pop")
#else
#define CLANG_IGNORE_IGNORED_ATTRIBUTES_PUSH
#define CLANG_IGNORE_IGNORED_ATTRIBUTES_POP
#endif