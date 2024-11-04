#pragma once

#include "common.h"

static const int MAX_RADIX_DEGREE = 8;
static const int MAX_LOCAL_WORK_SIZE_DEGREE = 7;

static void fr_fft(const int N, const int domN, storage* &gpu_poly, storage* &gpu_tmp_buffer, const storage* gpu_powers, 
                   const storage* gpu_omegas, const storage* gpu_pq, cudaStream_t st = 0)
{
    if (gpu_powers)
    {
        distribute_powers_kernel(gpu_poly, gpu_powers, N, st);
        if (N > 0)
            SAFE_CALL(cudaMemsetAsync(gpu_poly + N, 0, sizeof(storage) * (domN - N), st));
    }

    const int lgn = std::log2(domN);
    const int max_deg = std::min(MAX_RADIX_DEGREE, lgn);
    
    int lgp = 0;

    while (lgp < lgn)
    {
        int deg = std::min(max_deg, lgn - lgp);
        int lwsd = std::min(max_deg - 1, MAX_LOCAL_WORK_SIZE_DEGREE);

        int lws = 1 << lwsd;
        int gws = (domN >> 1) / lws;

        compute_fft_kernel(gpu_poly, gpu_tmp_buffer, gpu_pq, gpu_omegas, domN, lgp, deg, max_deg, gws, lws, st);
        std::swap(gpu_poly, gpu_tmp_buffer);
        lgp += deg;
    }
}

static void fr_ifft(const int N, storage* &gpu_poly, storage* &gpu_tmp_buffer, const storage* gpu_omegas_inv, 
                    const storage* gpu_pq_inv, const storage* gpu_powers, const storage& size_inv, cudaStream_t st = 0)
{
    const int lgn = std::log2(N);
    const int max_deg = std::min(MAX_RADIX_DEGREE, lgn);

    fr_fft(0, N, gpu_poly, gpu_tmp_buffer, nullptr, gpu_omegas_inv, gpu_pq_inv, st);
    scale(gpu_poly, size_inv, N, st);
    
    if (gpu_powers)
        distribute_powers_kernel(gpu_poly, gpu_powers, N, st);
}