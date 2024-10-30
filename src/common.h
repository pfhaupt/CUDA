#include <chrono>

#include "nob.h"

#ifdef DEBUG
#define PRINT_DEBUG(fmt, ...) \
    do { \
        printf("[DEBUG] "); \
        printf(fmt, __VA_ARGS__); \
    } while (0);
#else
#define PRINT_DEBUG(...)
#endif // DEBUG

#define ENSURE(what, descr) \
    do { \
        cudaError_t _err = what; \
        if (_err != cudaSuccess) { \
            printf("%s:%d: error: %s. %s\n", __FILE__, __LINE__, cudaGetErrorString(_err), descr); \
            exit(1); \
        } \
    } while (0);

#define hostCalloc(var, size, typ) \
    typ *var = (typ *) calloc(size, sizeof var[0]); \
    if (var == NULL) {\
        printf("%s:%d: error: Could not allocate memory for initializing %s.\n", __FILE__, __LINE__, #var); \
        exit(1); \
    } \
    nob_da_append(&host_ptrs, var); \
    PRINT_DEBUG("[HOST] Allocated %llu bytes for variable %s at addr %p.\n", sizeof(typ) * size, #var, var);

#define deviceCalloc(var, size, typ) \
    typ *var = NULL; \
    ENSURE(cudaMalloc(&var, size * sizeof var[0]), "Could not allocate memory on device"); \
    ENSURE(cudaMemset(var, size * sizeof var[0], 0), "Could not zero-initialize memory on device"); \
    if (var == NULL) {\
        printf("%s:%d: error: Could not allocate memory for initializing %s.\n", __FILE__, __LINE__, #var); \
        exit(1); \
    } \
    PRINT_DEBUG("[DEVICE] Allocated %llu bytes for variable %s at addr %p.\n", sizeof(typ) * size, #var, var);

#define verifyElementType(dst, src) \
    decltype(dst) [[maybe_unused]] _foo = src;

#define copyHostToDevice(dst, src, size) \
    do { \
        verifyElementType(dst, src); \
        ENSURE(cudaMemcpy(dst, src, size * sizeof src[0], cudaMemcpyHostToDevice), "Could not copy host memory to device"); \
        PRINT_DEBUG("Copied %llu bytes from host %p to device %p.\n", size * sizeof src[0], src, dst); \
    } while (0);

#define copyDeviceToHost(dst, src, size) \
    do { \
        verifyElementType(dst, src); \
        ENSURE(cudaMemcpy(dst, src, size * sizeof src[0], cudaMemcpyDeviceToHost), "Could not copy device memory to host"); \
        PRINT_DEBUG("Copied %llu bytes from device %p to host %p.\n", size * sizeof src[0], src, dst); \
    } while (0);

#define verifyMemory(dst, src, size) \
    do { \
        for (int i = 0; i < (size); i++) { \
            if (dst[i] != src[i]) { \
                printf("%s:%d: error: memory regions do not contain the same elements.\n", __FILE__, __LINE__); \
                printf("note: element mismatch at index %d\n", i); \
                printf("note: dst element: %d\n", dst[i]); \
                printf("note: src element: %d\n", src[i]); \
                exit(1); \
            } \
        } \
        PRINT_DEBUG("Memory regions %p and %p with size %d elements are the same.\n", dst, src, size); \
    } while (0); \

#define invokeKernel(kernel, gridDim, blockDim, ...) \
    do { \
        kernel<<<gridDim, blockDim>>>(__VA_ARGS__); \
        ENSURE(cudaGetLastError(), "Could not invoke kernel <"#kernel">"); \
    } while (0);

#define TIME(what, descr) \
    do { \
        auto start = std::chrono::high_resolution_clock::now(); \
        what; \
        auto end = std::chrono::high_resolution_clock::now(); \
        auto durMs = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); \
        printf("%s took %llu ms\n", descr, durMs); \
    } while (0);

#define TIME_AND_BIND(bindTo, what) \
    do { \
        auto start = std::chrono::high_resolution_clock::now(); \
        what; \
        auto end = std::chrono::high_resolution_clock::now(); \
        bindTo = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count(); \
    } while (0);


#define freeHost() \
    for (int i = 0; i < host_ptrs.count; i++) \
        free(host_ptrs.items[i]); \
    free(host_ptrs.items); \

#define freeDevice() \
    for (int i = 0; i < device_ptrs.count; i++) { \
        ENSURE(cudaFree(device_ptrs.items[i]), "Could not free memory on device"); \
    } \
    free(device_ptrs.items); \

#ifndef ELEM_TYPE
#define ELEM_TYPE int
#endif
typedef struct List {
    void **items;
    int count;
    int capacity;
} List;

List host_ptrs = {0};
List device_ptrs = {0};
