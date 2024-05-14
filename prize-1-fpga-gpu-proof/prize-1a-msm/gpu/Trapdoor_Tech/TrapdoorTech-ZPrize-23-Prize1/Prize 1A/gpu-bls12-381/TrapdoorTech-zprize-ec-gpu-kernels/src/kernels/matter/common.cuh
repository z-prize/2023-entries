#define HANDLE_CUDA_ERROR(statement)                                                                                                                           \
  {                                                                                                                                                            \
    cudaError_t hce_result = (statement); \
    printf("hce_result error :%d,__line:%d func:%s\n", hce_result,__LINE__, __func__);                                                                                                                         \
    if (hce_result != cudaSuccess)                                                                                                                             \
      return hce_result;                                                                                                                                       \
  }

#ifdef __CUDA_ARCH__
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif
