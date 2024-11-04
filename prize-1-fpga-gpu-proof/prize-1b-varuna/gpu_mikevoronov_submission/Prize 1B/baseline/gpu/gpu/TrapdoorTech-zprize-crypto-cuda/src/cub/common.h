#pragma once

#define SAFE_CALL(call) do { \
        cudaError err = call; \
        if (cudaSuccess != err) { \
                const char *errStr = cudaGetErrorString(err);\
                fprintf(stderr, "Cuda error %d in file '%s' in line %i : %s.\n", (int)err, __FILE__, __LINE__, errStr); \
                exit(-1); \
                } \
} while (0)
    
static void register_host(const void* ptr, const int N)
{
    if (ptr && N)
        SAFE_CALL(cudaHostRegister ((void*)ptr, sizeof(storage) * N, cudaHostRegisterDefault));
}

template<typename T>
static void duplicate(T* gpu_p1, const T* gpu_p2, const int N, cudaStream_t st = 0)
{    
    if (gpu_p1 && gpu_p2 && N)
        SAFE_CALL(cudaMemcpyAsync(gpu_p1, gpu_p2, sizeof(T) * N, cudaMemcpyDeviceToDevice, st));
}

template<typename T>
static void copy_to_device(T* gpu_p, const T* p, const int N, cudaStream_t st = 0)
{ 
    if (gpu_p && p && N)
        SAFE_CALL(cudaMemcpyAsync(gpu_p, p, sizeof(T) * N, cudaMemcpyHostToDevice, st));
}

template<typename T>
static void copy_to_host(const T* gpu_p, T* p, const int N, cudaStream_t st = 0)
{
    if (gpu_p && p && N)
        SAFE_CALL(cudaMemcpyAsync(p, gpu_p, sizeof(storage) * N, cudaMemcpyDeviceToHost, st));
}

static int cores = 0;
static int getCudaCores() 
{
    if (cores == 0) 
    {
        struct cudaDeviceProp props;
        cudaGetDeviceProperties(&props, 0);
        
        cores = props.multiProcessorCount;
    }
    return cores;
}

static void setSizesForInverse(int& spawn, int& chunk_size, int& bucket_num, const int N)
{
    int max_threads = get_invesion_blocks_count() * 32;
    const int total_cores = getCudaCores() * max_threads / 4; //TODO: need to profile!!
    //printf("max_threads = %d, total_cores %d\n", max_threads, total_cores);

    const int MAX_BUCKET_SIZE = 6;
    const int MAX_BUCKET_NUM = 1 << MAX_BUCKET_SIZE;
    
    spawn = std::min(N, total_cores);
    chunk_size = N / spawn + (N % spawn != 0);
    bucket_num = std::min(chunk_size / MAX_BUCKET_SIZE + (chunk_size % MAX_BUCKET_SIZE != 0), MAX_BUCKET_NUM);
}

static void setSizesForDivide(int& chunk_size, int& chunk_size1, int& bucket_num, int& deep, const int N)
{
    int max_threads = 1024; //TODO
    const int total_cores = getCudaCores() * max_threads;
        
    bucket_num = std::min(N, total_cores);
    chunk_size = N / bucket_num + (N % bucket_num != 0);
    chunk_size = std::max(2, chunk_size);
    
    
    chunk_size1 = 4;
        
    int nextN = N / chunk_size + ((N % chunk_size) != 0);
    bucket_num = nextN;

    deep = 1;
    int tmp = chunk_size1;
    while (tmp < nextN)
    {
        deep++;
        tmp *= chunk_size1;
    }
}