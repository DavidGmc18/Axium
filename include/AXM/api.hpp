#pragma once

#if defined(_WIN32) && defined(AXIUM_EXPORTS)
    #  define AXM_API __declspec(dllexport)
#elif defined(_WIN32)
    #  define AXM_API __declspec(dllimport)
#else
    #  define AXM_API
#endif