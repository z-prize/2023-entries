#pragma once
#include <cstdio>
#include <cuda_runtime.h>

#define HANDLE_CUDA_ERROR(statement)                                                                                                                           \
  {                                                                                                                                                            \
    cudaError_t hce_result = (statement); \
    printf("hce_result error :%d,__line:%d func:%s\n", hce_result,__LINE__, __func__);                                                                                                                         \
    if (hce_result != cudaSuccess)                                                                                                                             \
      return hce_result;                                                                                                                                       \
  }

#define SAFE_CALL(call) do { \
        cudaError err = call; \
        if (cudaSuccess != err) { \
                const char *errStr = cudaGetErrorString(err);\
                fprintf(stderr, "Cuda error %d in file '%s' in line %i : %s.\n", (int)err, __FILE__, __LINE__, errStr); \
                exit(-1); \
                } \
} while (0)

#ifdef __CUDA_ARCH__
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif
