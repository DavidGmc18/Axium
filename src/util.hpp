#pragma once

#define ABORT() {fprintf(stderr, "Abort at %s:%d\n", __FILE__, __LINE__); std::abort();}

#ifdef DEBUG
    #define ABORT_DEBUG() ABORT()
#else
    #define ABORT_DEBUG()
#endif
 

#define PTR_CHECK(ptr) if (ptr == nullptr) {std::cerr << "Nullptr!\n"; ABORT_DEBUG()}