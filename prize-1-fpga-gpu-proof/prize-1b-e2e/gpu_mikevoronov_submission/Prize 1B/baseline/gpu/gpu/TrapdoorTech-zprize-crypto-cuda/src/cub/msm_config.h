#include <stddef.h>
#include <cuda_runtime.h>
#include "msm_config.cuh"

extern "C" {
#if MSM_YRRID
	void*   createContext(uint32_t curve, uint32_t maxBatchCount, uint32_t maxPointCount);
	int32_t destroyContext(void* contextPtr);
	int32_t preprocessOnlyPoints(void* contextPtr, void* pointData, uint32_t pointCount, void* preprocessedContext);
	int32_t process(void* contextPtr, void* resultData, void* scalarData, uint32_t pointCount, void* tmpResult, cudaStream_t runStream);
#else
    void bls12_381_left_shift_kernel(void *values, const unsigned shift, const unsigned count, cudaStream_t st = 0);
    void execute_msm(MsmConfiguration cfg, bool get_memory = false);
#endif
}

static void do_msm(const std::vector<void*>& scalars, int call_count, cudaStream_t st, bool get_memory);

template<int T>
static void do_msm_yrrid(const std::vector<void*> &scalars_v, int call_count, void* tmp_pool, int N, cudaStream_t st);