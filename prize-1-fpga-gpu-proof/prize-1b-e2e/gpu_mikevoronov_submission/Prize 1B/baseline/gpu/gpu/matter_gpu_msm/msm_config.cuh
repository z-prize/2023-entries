#pragma once

#define BLS_381_FR_MODULUS_BITS 255

struct MsmConfiguration {
    void *bases;
    void *scalars;
    void *results;
    void *results_tmp;
    int log_scalars_count;
    
    bool force_min_chunk_size;
    int log_min_chunk_size;
    bool force_max_chunk_size;
    int log_max_chunk_size;
    int window_bits_count;
    int precomputed_windows_stride;
    int precomputed_bases_stride;
    bool scalars_not_montgomery;
    
    int log_min_inputs_count;
    int log_max_inputs_count;
    int multiprocessor_count;
    cudaStream_t st;
};
        
struct MSMContext {
    int log_count;
    void *bases = nullptr;
    int window_bits_count;
    int precompute_factor;   
    void* result = nullptr;
	void* result_affine = nullptr;
    void** result_tmp = nullptr;
    
    MSMContext(int comms) {
        result_tmp = new void*[comms];
    }
};