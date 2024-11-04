#include <thread>
#include <vector>
#include <chrono>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <map>
#include <stack>

#define MSM_YRRID 1

#include "poly_functions.h"

#include "msm_config.h"
#include "types.h"
#include "prover_key.h"
#include "debug.h"
#include "memory.h"
#include "poseidon_hash.h"
#include "fft.h"

using namespace std;

static int msm_call = 0;

static cudaStream_t execSt1 = 0;
static cudaStream_t execSt2 = 0;
static cudaStream_t execSt3 = 0;
static cudaStream_t execSt4 = 0;
static cudaStream_t execSt5 = 0;

static const int COMMITMENTS = 13;

static std::thread* compute_w;

#if MSM_YRRID
static void *MSMContextYrrid[2] = { nullptr, nullptr};
#endif
static MSMContext msm_context(COMMITMENTS);

static storage* gpu_w_l_sc = nullptr;
static storage* gpu_w_r_sc = nullptr;
static storage* gpu_w_o_sc  = nullptr;
static storage* gpu_w_4_sc  = nullptr;

static storage* gpu_w_l_poly = nullptr;
static storage* gpu_w_r_poly = nullptr;
static storage* gpu_w_o_poly = nullptr;
static storage* gpu_w_4_poly = nullptr;

static storage* gpu_w_l_poly_ex = nullptr;
static storage* gpu_w_r_poly_ex = nullptr;
static storage* gpu_w_o_poly_ex = nullptr;
static storage* gpu_w_4_poly_ex = nullptr;

static storage* gpu_omegas = nullptr;
static storage* gpu_omegas_8n = nullptr;
static storage* gpu_omegas_inv = nullptr;
static storage* gpu_omegas_inv8n = nullptr;

static storage* gpu_pq = nullptr;
static storage* gpu_pq_inv = nullptr;
static storage* gpu_pq_8n = nullptr;
static storage* gpu_pq_inv8n = nullptr;

static storage* gpu_powers = nullptr;
static storage* gpu_quotient = nullptr;
static storage* gpu_linear = nullptr;

static storage* gpu_pi_poly = nullptr;
static storage* gpu_pi_poly_ex = nullptr;

static storage* pi_from_gpu = nullptr;
static storage* gpu_z_poly = nullptr;    

static vector<pair<int, storage>> pi;



static void inverse_poly(storage* gpu_poly, int N, cudaStream_t st = 0) 
{
    int spawn, chunk_size, bucket_num;        
    setSizesForInverse(spawn, chunk_size, bucket_num, N);

    const int lws = 32;
    const int gws = spawn / lws + ((spawn % lws) != 0);

    storage* gpu_tmp_inv = get_pool(3 * spawn + 1);
    storage* gpu_buckects = get_pool(N);
    
    invesion_kernel(gpu_poly, gpu_tmp_inv, gpu_buckects, spawn, gws, lws, chunk_size, bucket_num, N, st);
    release_pool(gpu_tmp_inv, 3 * spawn + 1);
    release_pool(gpu_buckects, N);
}

static void compute_w_polys(const void *omegas, //omegas + omegas8n
                            const void *omegas_inv, //omegas_inv + omegas_inv8n
                            const void *size_inv, // size_invN + size_inv8N
                            const int N)
{
    gpu_w_l_poly = get_pool(N);
    gpu_w_r_poly = get_pool(N);
    gpu_w_o_poly = get_pool(N);
    gpu_w_4_poly = get_pool(N);

    perm_gpu.allocate_sigma(N, 0);
    
    storage* gpu_tmp_buffer = get_pool(N);
    storage* gpu_tmp_buffer1 = get_pool(N);

    duplicate(gpu_w_l_poly, gpu_w_l_sc, N, execSt1);
    fr_ifft(N, gpu_w_l_poly, gpu_tmp_buffer, gpu_omegas_inv, gpu_pq_inv, nullptr, ((storage*)size_inv)[0], execSt1);
    
    duplicate(gpu_w_r_poly, gpu_w_r_sc, N, execSt1);
    fr_ifft(N, gpu_w_r_poly, gpu_tmp_buffer, gpu_omegas_inv, gpu_pq_inv, nullptr, ((storage*)size_inv)[0], execSt1);
    
    duplicate(gpu_w_o_poly, gpu_w_o_sc, N, execSt2);
    fr_ifft(N, gpu_w_o_poly, gpu_tmp_buffer1, gpu_omegas_inv, gpu_pq_inv, nullptr, ((storage*)size_inv)[0], execSt2);
    
    duplicate(gpu_w_4_poly, gpu_w_4_sc, N, execSt2);
    fr_ifft(N, gpu_w_4_poly, gpu_tmp_buffer1, gpu_omegas_inv, gpu_pq_inv, nullptr, ((storage*)size_inv)[0], execSt2);
    
    duplicate(perm_gpu.left_sigma[0], prover_gpu.perm.left_sigma[0], N, execSt1);
    fr_fft(0, N, perm_gpu.left_sigma[0], gpu_tmp_buffer, nullptr, gpu_omegas, gpu_pq, execSt1);

    duplicate(perm_gpu.right_sigma[0], prover_gpu.perm.right_sigma[0], N, execSt2);
    fr_fft(0, N, perm_gpu.right_sigma[0], gpu_tmp_buffer1, nullptr, gpu_omegas, gpu_pq, execSt2);

    duplicate(perm_gpu.out_sigma[0], prover_gpu.perm.out_sigma[0], N, execSt1);
    fr_fft(0, N, perm_gpu.out_sigma[0], gpu_tmp_buffer, nullptr, gpu_omegas, gpu_pq, execSt1);

    duplicate(perm_gpu.fourth_sigma[0], prover_gpu.perm.fourth_sigma[0], N, execSt2);
    fr_fft(0, N, perm_gpu.fourth_sigma[0], gpu_tmp_buffer1, nullptr, gpu_omegas, gpu_pq, execSt2); 
    
#if MSM_YRRID
    auto msm_th1 = std::thread(do_msm_yrrid<0>, vector<void*>{gpu_w_l_poly, gpu_w_r_poly}, msm_call, gpu_tmp_buffer, N, execSt1);
    msm_call += 2;
    auto msm_th2 = std::thread(do_msm_yrrid<1>, vector<void*>{gpu_w_o_poly, gpu_w_4_poly}, msm_call, gpu_tmp_buffer1, N, execSt2);
    msm_call += 2;
#else
    auto msm_th1 = std::thread(do_msm, vector<void*>{gpu_w_l_poly, gpu_w_r_poly}, msm_call, execSt1, false);
    msm_call += 2;
    auto msm_th2 = std::thread(do_msm, vector<void*>{gpu_w_o_poly, gpu_w_4_poly}, msm_call, execSt2, false);
    msm_call += 2;
#endif

    msm_th1.join();
    msm_th2.join();
    
    SAFE_CALL(cudaDeviceSynchronize());
    release_pool(gpu_tmp_buffer, N);
    release_pool(gpu_tmp_buffer1, N);
    release_pool(gpu_pq, N);
}
    
extern "C" {
    int get_commit(const int idx, void *commitRust, void *commitRust_affine)
    {
		int is_affine = 0;
		if (MSM_YRRID) {
            memset(commitRust_affine, 0, sizeof(affineG1));            
			memcpy(commitRust_affine, (affine*)msm_context.result_affine + idx, sizeof(affine));

			is_affine = 1;
        }
		else
			memcpy(commitRust, (point_jacobian*)msm_context.result + idx, sizeof(point_jacobian));

		return is_affine;
    }
    
    void compute_scalars(const void *omegas, //omegas + omegas8n
                         const void *omegas_inv, //omegas_inv + omegas_inv8n
                         const void *size_inv, // size_invN + size_inv8N
                         const int N) {

        gpu_w_l_sc = get_pool(N);
        gpu_w_r_sc = get_pool(N);
        gpu_w_o_sc = get_pool(N);
        gpu_w_4_sc = get_pool(N);

        gpu_omegas_inv = get_pool(N);
        gpu_pq_inv = get_pool(N);
        gpu_omegas = get_pool(N);
        gpu_pq = get_pool(N);
        
        SAFE_CALL(cudaMemsetAsync(gpu_w_l_sc, 0, sizeof(storage) * N, execSt3));
        SAFE_CALL(cudaMemsetAsync(gpu_w_r_sc, 0, sizeof(storage) * N, execSt3));
        SAFE_CALL(cudaMemsetAsync(gpu_w_o_sc, 0, sizeof(storage) * N, execSt3));
        SAFE_CALL(cudaMemsetAsync(gpu_w_4_sc, 0, sizeof(storage) * N, execSt3));
        
        gpu_pi_poly = (storage*)build_constraints_on_gpu(gpu_w_l_sc, gpu_w_r_sc, gpu_w_o_sc, gpu_w_4_sc, execSt3);
        
        compute_pq_kernel(gpu_pq_inv, ((const storage*)omegas_inv)[0], std::min(MAX_RADIX_DEGREE, (int)std::log2(N)), N, execSt1);
        compute_pq_kernel(gpu_pq, ((const storage*)omegas)[0], std::min(MAX_RADIX_DEGREE, (int)std::log2(N)), N, execSt1);
        
        get_powers_kernel(gpu_omegas_inv, ((const storage*)omegas_inv)[0], N, execSt2);
        get_powers_kernel(gpu_omegas, ((const storage*)omegas)[0], N, execSt2);
        SAFE_CALL(cudaDeviceSynchronize());
    }

    void start_compute_w_polys(const void *omegas, //omegas + omegas8n
                               const void *omegas_inv, //omegas_inv + omegas_inv8n
                               const void *size_inv, // size_invN + size_inv8N
                               const int N)
    {
        compute_w = new std::thread(compute_w_polys, omegas, omegas_inv, size_inv, N);
    }
    
    void wait_compute_w_polys() 
    {
        compute_w->join();
    }

    int calculate_pi_size(int N) {
        copy_to_host(gpu_pi_poly, pi_from_gpu, N, execSt3);
        SAFE_CALL(cudaStreamSynchronize(execSt3));
        
        const auto zero = fd_q::get_zero();
        int count = 0;
        for (int z = 0; z < N; ++z) {
            if (pi_from_gpu[z] != zero) {
                count++;
                pi.push_back(std::make_pair(z, pi_from_gpu[z]));
            }
        }
        return count;
    }
    
    void get_pi_pos(void* pos_, void* items_) {
        size_t* pos = (size_t *)pos_;
        storage* items = (storage*)items_;

        for (int z = 0; z < pi.size(); ++z) {
            pos[z] = pi[z].first;
            items[z] = pi[z].second;
        }
    }

    void compute_z_poly (const void *beta,
                         const void *gamma,
                         const void *K1, //K1 * beta
                         const void *K2, //K2 * beta
                         const void *K3, //K3 * beta
                         const void *size_inv, // size_invN + size_inv8N
                         const void *g, // g + g_inv 
                         const void *omegas, //omegas + omegas8n
                         const int N) 
    {
        const int domN = N * 8;//TODO move to parameter

        storage* gpu_inverse = get_pool(N);
        compute_z_poly_denum_kernel(gpu_inverse, gpu_w_l_sc, gpu_w_r_sc, gpu_w_o_sc, gpu_w_4_sc, 
                                    perm_gpu.left_sigma[0], perm_gpu.right_sigma[0], perm_gpu.out_sigma[0], perm_gpu.fourth_sigma[0],
                                    ((storage*)beta)[0], ((storage*)gamma)[0], N, execSt1);
        
        inverse_poly(gpu_inverse, N, execSt1);
        perm_gpu.release_data(N, 0);

        gpu_z_poly = get_pool(N);
        const storage* gpu_domain_roots = gpu_omegas;
        compute_z_poly_kernel(gpu_z_poly, gpu_inverse, gpu_w_l_sc, gpu_w_r_sc, gpu_w_o_sc, gpu_w_4_sc, gpu_domain_roots, 
                              ((storage*)K1)[0], ((storage*)K2)[0], ((storage*)K3)[0], ((storage*)beta)[0], ((storage*)gamma)[0], N, execSt1);

        storage *gpu_tmp = get_pool(N);
        fr_ifft(N, gpu_z_poly, gpu_tmp, gpu_omegas_inv, gpu_pq_inv, nullptr, ((storage*)size_inv)[0], execSt1);
        SAFE_CALL(cudaStreamSynchronize(execSt1));

#if MSM_YRRID		
		storage* gpu_msm = get_pool(N);
		auto msm_th = std::thread(do_msm_yrrid<0>, vector<void*>{gpu_z_poly}, msm_call++, gpu_msm, N, execSt2);
#else
        auto msm_th = std::thread(do_msm, vector<void*>{gpu_z_poly}, msm_call++, execSt2, false);
#endif
        fr_ifft(N, gpu_pi_poly, gpu_tmp, gpu_omegas_inv, gpu_pq_inv, nullptr, ((storage*)size_inv)[0], execSt1);

        release_pool(gpu_tmp, N);
        release_pool(gpu_omegas_inv, N);
        release_pool(gpu_pq_inv, N);
        release_pool(gpu_inverse, N);
        release_pool(gpu_omegas, N);
        
        gpu_powers = get_pool(N);
        gpu_pq_8n = get_pool(N);
        
        gpu_omegas_8n = get_pool(domN);
        gpu_w_l_poly_ex = get_pool(domN);
        gpu_w_r_poly_ex = get_pool(domN);
        gpu_w_o_poly_ex = get_pool(domN);
        gpu_w_4_poly_ex = get_pool(domN);
        storage* gpu_tmp_buffer = get_pool(domN);
        
        get_powers_kernel(gpu_powers, ((storage*)g)[0], N, execSt1);        
        compute_pq_kernel(gpu_pq_8n, ((const storage*)omegas)[1], std::min(MAX_RADIX_DEGREE, (int)std::log2(domN)), domN, execSt1);
        get_powers_kernel(gpu_omegas_8n, ((const storage*)omegas)[1], domN, execSt1);
        
        duplicate(gpu_w_l_poly_ex, gpu_w_l_poly, N, execSt1);
        fr_fft(N, domN, gpu_w_l_poly_ex, gpu_tmp_buffer, gpu_powers, gpu_omegas_8n, gpu_pq_8n, execSt1);
        
        duplicate(gpu_w_r_poly_ex, gpu_w_r_poly, N, execSt1);
        fr_fft(N, domN, gpu_w_r_poly_ex, gpu_tmp_buffer, gpu_powers, gpu_omegas_8n, gpu_pq_8n, execSt1);
        
        duplicate(gpu_w_o_poly_ex, gpu_w_o_poly, N, execSt1);
        fr_fft(N, domN, gpu_w_o_poly_ex, gpu_tmp_buffer, gpu_powers, gpu_omegas_8n, gpu_pq_8n, execSt1);
        
        duplicate(gpu_w_4_poly_ex, gpu_w_4_poly, N, execSt1);
        fr_fft(N, domN, gpu_w_4_poly_ex, gpu_tmp_buffer, gpu_powers, gpu_omegas_8n, gpu_pq_8n, execSt1);

        gpu_pi_poly_ex = get_pool(domN);
        duplicate(gpu_pi_poly_ex, gpu_pi_poly, N, execSt1);
        fr_fft(N, domN, gpu_pi_poly_ex, gpu_tmp_buffer, gpu_powers, gpu_omegas_8n, gpu_pq_8n, execSt1);

        msm_th.join();
        SAFE_CALL(cudaStreamSynchronize(execSt2));
        
        release_pool(gpu_tmp_buffer, domN);
        release_pool(gpu_pi_poly, N);
#if MSM_YRRID
		release_pool(gpu_msm, N);		
#endif
    }
    
    void compute_quotient (int N, int domN, 

                           const void *g, // g + g_inv 
                           const void *alpha,
                           const void *alpha_sq,
                           const void *beta,
                           const void *gamma,
                           const void *size_inv, // size_invN + size_inv8N
                           const void *K1, //K1 * beta
                           const void *K2, //K2 * beta
                           const void *K3, //K3 * beta
                           
                           const void *omegas, //omegas + omegas8n
                           const void *omegas_inv //omegas_inv + omegas_inv8n
                           ) 
    {        
        storage* gpu_gate_constraints = get_pool(domN);
        compute_gate_constraint_kernel   (gpu_gate_constraints, 
                                          gpu_w_l_poly_ex, 
                                          gpu_w_r_poly_ex,
                                          gpu_w_o_poly_ex,
                                          gpu_w_4_poly_ex,

                                          prover_gpu.arith.q_m[1],
                                          prover_gpu.arith.q_l[1],
                                          prover_gpu.arith.q_r[1],
                                          prover_gpu.arith.q_o[1],
                                          prover_gpu.arith.q_4[1],
                                          prover_gpu.arith.q_hl[1],
                                          prover_gpu.arith.q_hr[1],
                                          prover_gpu.arith.q_h4[1],
                                          prover_gpu.arith.q_c[1],
                                          prover_gpu.arith.q_arith[1],
                                           
                                          gpu_pi_poly_ex,
                                          domN, execSt1);

        prover_gpu.arith.release_data(prover_gpu.N, 1);
        release_pool(gpu_pi_poly_ex, domN);

        gpu_omegas_inv8n = get_pool(domN);
        gpu_pq_inv8n = get_pool(N);

        get_powers_kernel(gpu_omegas_inv8n, ((const storage*)omegas_inv)[1], domN, execSt1);
        compute_pq_kernel(gpu_pq_inv8n, ((const storage*)omegas_inv)[1], std::min(MAX_RADIX_DEGREE, (int)std::log2(domN)), domN, execSt1);

        storage* gpu_l1_alpha = get_pool(domN);
        SAFE_CALL(cudaMemsetAsync(gpu_l1_alpha, 0, sizeof(storage) * N, execSt1));
        set_value(gpu_l1_alpha, ((const storage*)alpha_sq)[0], 0, execSt1);
        
        storage* gpu_tmp_buffer = get_pool(domN);
        fr_ifft(N, gpu_l1_alpha, gpu_tmp_buffer, gpu_omegas_inv8n, gpu_pq_inv8n, nullptr, ((storage*)size_inv)[0], execSt1);
        fr_fft(N, domN, gpu_l1_alpha, gpu_tmp_buffer, gpu_powers, gpu_omegas_8n, gpu_pq_8n, execSt1);
                
        storage* gpu_z_poly_ex = get_pool(domN);
        duplicate(gpu_z_poly_ex, gpu_z_poly, N, execSt1);
        fr_fft(N, domN, gpu_z_poly_ex, gpu_tmp_buffer, gpu_powers, gpu_omegas_8n, gpu_pq_8n, execSt1);

        release_pool(gpu_tmp_buffer, domN);
        release_pool(gpu_omegas_8n, domN);
        release_pool(gpu_pq_8n, N);
        release_pool(gpu_powers, N);

        storage* gpu_v_h_inv = get_pool(domN);
        duplicate(gpu_v_h_inv, prover_gpu.v_h, domN, execSt1);

        storage* gpu_permutation = get_pool(domN);
        compute_permutation_kernel   (gpu_permutation,
                                      gpu_w_l_poly_ex, 
                                      gpu_w_r_poly_ex,
                                      gpu_w_o_poly_ex,
                                      gpu_w_4_poly_ex,
                                      gpu_z_poly_ex,
                                      prover_gpu.perm.linear_eval,
                                      gpu_l1_alpha,

                                      prover_gpu.perm.left_sigma[1],
                                      prover_gpu.perm.right_sigma[1],
                                      prover_gpu.perm.out_sigma[1],
                                      prover_gpu.perm.fourth_sigma[1],
                                      
                                      ((storage*)alpha)[0], ((storage*)beta)[0], ((storage*)gamma)[0],
                                      ((storage*)K1)[0], ((storage*)K2)[0], ((storage*)K3)[0],
                                      domN, execSt1);

        prover_gpu.perm.release_data(prover_gpu.N, 1);

        release_pool(gpu_l1_alpha, domN);

        release_pool(gpu_w_l_poly_ex, domN);
        release_pool(gpu_w_r_poly_ex, domN);
        release_pool(gpu_w_o_poly_ex, domN);
        release_pool(gpu_w_4_poly_ex, domN);
        
        release_pool(gpu_z_poly_ex, domN);

        inverse_poly(gpu_v_h_inv, domN, execSt1);
        
        gpu_quotient = get_pool(domN);
        compute_quotient_kernel(gpu_quotient, gpu_gate_constraints, gpu_permutation, gpu_v_h_inv, domN, execSt1);
        
        prover_gpu.release_data();

        storage* gpu_powers_inv = get_pool(domN);
        get_powers_kernel(gpu_powers_inv, ((storage*)g)[1], domN, execSt1);
        gpu_tmp_buffer = get_pool(domN);
        fr_ifft(domN, gpu_quotient, gpu_tmp_buffer, gpu_omegas_inv8n, gpu_pq_inv8n, gpu_powers_inv, ((storage*)size_inv)[1], execSt1);
        release_pool(gpu_tmp_buffer, domN);
        
        release_pool(gpu_omegas_inv8n, domN);
        release_pool(gpu_pq_inv8n, N);
        
        release_pool(gpu_gate_constraints, domN);        
        release_pool(gpu_permutation, domN);
        release_pool(gpu_powers_inv, domN);
        release_pool(gpu_v_h_inv, domN);
        
        SAFE_CALL(cudaStreamSynchronize(execSt1));

#if MSM_YRRID
        storage* gpu_msm1 = get_pool(N);
        storage* gpu_msm2 = get_pool(N);

        auto msm_th1 = std::thread(do_msm_yrrid<0>, vector<void*>{gpu_quotient, gpu_quotient + N, gpu_quotient + 2*N}, msm_call, gpu_msm1, N, execSt1);
        msm_call += 3;
        auto msm_th2 = std::thread(do_msm_yrrid<1>, vector<void*>{gpu_quotient + 3*N, gpu_quotient + 4*N, gpu_quotient + 5*N}, msm_call, gpu_msm2, N, execSt2);
        msm_call += 3;

        msm_th1.join();
        msm_th2.join();
		SAFE_CALL(cudaDeviceSynchronize());
		release_pool(gpu_msm1, N);
		release_pool(gpu_msm2, N);
#else
        auto msm_th1 = std::thread(do_msm, vector<void*>{gpu_quotient, gpu_quotient + N, gpu_quotient + 2*N}, msm_call, execSt1, false);
        msm_call += 3;
        auto msm_th2 = std::thread(do_msm, vector<void*>{gpu_quotient + 3*N, gpu_quotient + 4*N, gpu_quotient + 5*N}, msm_call, execSt2, false);
        msm_call += 3;
        
        msm_th1.join();
        msm_th2.join();
        SAFE_CALL(cudaDeviceSynchronize());
#endif
    }

    void compute_linearisation_poly(void* values, // zeta + sh_zeta + vanishing_poly_eval + z_challenge_to_n + alpha + beta + gamma + l1_alpha_sq;                                    
                                    const int N) 
    {
        storage* powers = get_pool(N);
        storage* powers_shift = get_pool(N);
        
        get_powers_kernel(powers, ((storage*)values)[0], N);
        get_powers_kernel(powers_shift, ((storage*)values)[1], N);
        
        storage a_eval, b_eval, c_eval, d_eval;
        storage left_sigma_eval, right_sigma_eval, out_sigma_eval;
        storage permutation_eval, a_next_eval, b_next_eval, d_next_eval;
        storage q_arith_eval, q_c_eval, q_l_eval, q_r_eval;
        storage q_hl_eval, q_hr_eval, q_h4_eval;

        evaluate_kernel(gpu_w_l_poly, powers, a_eval, N, execSt1);
        evaluate_kernel(gpu_w_r_poly, powers, b_eval, N, execSt2);
        evaluate_kernel(gpu_w_o_poly, powers, c_eval, N, execSt3);
        evaluate_kernel(gpu_w_4_poly, powers, d_eval, N, execSt1);
        
        evaluate_kernel(prover_gpu.perm.left_sigma[0], powers, left_sigma_eval, N, execSt2);
        evaluate_kernel(prover_gpu.perm.right_sigma[0], powers, right_sigma_eval, N, execSt3);
        evaluate_kernel(prover_gpu.perm.out_sigma[0], powers, out_sigma_eval, N, execSt1);
        evaluate_kernel(gpu_z_poly, powers_shift, permutation_eval, N, execSt2);
        evaluate_kernel(prover_gpu.arith.q_arith[0], powers, q_arith_eval, N, execSt3);
        evaluate_kernel(prover_gpu.arith.q_c[0], powers, q_c_eval, N, execSt1);
        evaluate_kernel(prover_gpu.arith.q_l[0], powers, q_l_eval, N, execSt2);
        evaluate_kernel(prover_gpu.arith.q_r[0], powers, q_r_eval, N, execSt3);        
        evaluate_kernel(prover_gpu.arith.q_hl[0], powers, q_hl_eval, N, execSt1);
        evaluate_kernel(prover_gpu.arith.q_hr[0], powers, q_hr_eval, N, execSt2);
        evaluate_kernel(prover_gpu.arith.q_h4[0], powers, q_h4_eval, N, execSt3);
        evaluate_kernel(gpu_w_l_poly, powers_shift, a_next_eval, N, execSt1);
        evaluate_kernel(gpu_w_r_poly, powers_shift, b_next_eval, N, execSt2);
        evaluate_kernel(gpu_w_4_poly, powers_shift, d_next_eval, N, execSt3);

        SAFE_CALL(cudaDeviceSynchronize());

        storage* gpu_quotient_term = get_pool(N);
        compress_quotient_term_kernel(gpu_quotient_term, gpu_quotient, ((storage*)values)[2], ((storage*)values)[3], N);
        
        storage* gpu_buffer = get_pool(N);        
        storage* gpu_perm = get_pool(N);
        compute_linearisation_kernel   (gpu_perm, 
                                        gpu_buffer,
                                        gpu_z_poly, 
                                        prover_gpu.perm.fourth_sigma[0], 

                                        ((storage*)values)[0],
                                        ((storage*)values)[4],
                                        ((storage*)values)[5],
                                        ((storage*)values)[6],
                                        a_eval, b_eval, c_eval, d_eval,
                                        left_sigma_eval, right_sigma_eval, out_sigma_eval,
                                        permutation_eval,
                                        ((storage*)values)[7], N);

        gpu_linear = get_pool(N);
        compute_linearisation_kernel(gpu_linear, gpu_perm, gpu_quotient_term,
                                    prover_gpu.arith.q_m[0], 
                                    prover_gpu.arith.q_l[0], 
                                    prover_gpu.arith.q_r[0],
                                    prover_gpu.arith.q_o[0],
                                    prover_gpu.arith.q_4[0], 
                                    prover_gpu.arith.q_hl[0], 
                                    prover_gpu.arith.q_hr[0], 
                                    prover_gpu.arith.q_h4[0],
                                    prover_gpu.arith.q_c[0],
                                    a_eval, b_eval, c_eval, d_eval, q_arith_eval, N);
        
        SAFE_CALL(cudaDeviceSynchronize());
        
        prover_gpu.arith.release_data(prover_gpu.N, 0);

        release_pool(gpu_perm, N);
        release_pool(gpu_quotient_term, N);
        release_pool(gpu_buffer ,N);
        release_pool(powers, N);
        release_pool(powers_shift, N);
                
        storage* host_values = (storage*)values;
        host_values[0] = a_eval;
        host_values[1] = b_eval;
        host_values[2] = c_eval;
        host_values[3] = d_eval;
        host_values[4] = left_sigma_eval;
        host_values[5] = right_sigma_eval;
        host_values[6] = out_sigma_eval;
        host_values[7] = permutation_eval;
        host_values[8] = a_next_eval;
        host_values[9] = b_next_eval;
        host_values[10] = d_next_eval;
        host_values[11] = q_arith_eval;
        host_values[12] = q_c_eval;
        host_values[13] = q_l_eval;
        host_values[14] = q_r_eval;
        host_values[15] = q_hl_eval;
        host_values[16] = q_hr_eval;
        host_values[17] = q_h4_eval;
    }
   
    void compute_open_commits(void* values,// zeta, aw_ch, zeta*domain(1), sw_ch
                              const int N)
    {
        storage* gpu_comb_aw = get_pool(N);
        storage* gpu_comb_aw1 = get_pool(N);
        
        storage* gpu_comb_sw = get_pool(N);
        storage* gpu_comb_sw1 = get_pool(N);
        
        storage* gpu_powers1 = get_pool(N);
        storage* gpu_powers2 = get_pool(N);
        
        int chunk_size, buckets_num, chunk_size1, deep;
        setSizesForDivide(chunk_size, chunk_size1, buckets_num, deep, N);
        const int nextN = N / chunk_size + 1;

        //printf("ch %d, bm %d, N %d, deep %d, nextN %d\n", chunk_size, buckets_num, N, deep, nextN);
        std::vector<storage*> gpu_tmp1;
        std::vector<storage*> gpu_tmp2;
        int div = 1;
        for (int z = 0; z < deep + 1; ++z) {
            gpu_tmp1.push_back(get_pool(nextN / div));
            gpu_tmp2.push_back(get_pool(nextN / div));
            
            div *= 2;
        }
        
        compute_combination_aw_kernel (gpu_comb_aw, gpu_linear, 
                                       prover_gpu.perm.left_sigma[0],
                                       prover_gpu.perm.right_sigma[0],
                                       prover_gpu.perm.out_sigma[0],
                                       gpu_w_l_poly,
                                       gpu_w_r_poly,
                                       gpu_w_o_poly,
                                       gpu_w_4_poly,
                                       ((storage*)values)[1],
                                       N, execSt1);
        
        compute_poly_division_kernel(gpu_comb_aw, gpu_comb_aw1, gpu_tmp1.data(), gpu_powers1, ((storage*)values)[0], 
                                      buckets_num, chunk_size, deep, chunk_size1, N, execSt1);        
#if !MSM_YRRID
        auto msm_th1 = std::thread(do_msm, vector<void*>{gpu_comb_aw1}, msm_call++, execSt1, false);
#endif
        compute_combination_ws_kernel(gpu_comb_sw, 
                                      gpu_z_poly, 
                                      gpu_w_l_poly,
                                      gpu_w_r_poly,
                                      gpu_w_4_poly,
                                      ((storage*)values)[3],
                                      N, execSt2);
        
        compute_poly_division_kernel(gpu_comb_sw, gpu_comb_sw1, gpu_tmp2.data(), gpu_powers2, ((storage*)values)[2], 
                                      buckets_num, chunk_size, deep, chunk_size1, N, execSt2);
#if !MSM_YRRID
        auto msm_th2 = std::thread(do_msm, vector<void*>{gpu_comb_sw1}, msm_call++, execSt2, false);
		
        msm_th1.join();
        msm_th2.join();
        SAFE_CALL(cudaDeviceSynchronize());
#endif

#if MSM_YRRID
		storage* gpu_msm1 = get_pool(N);
		storage* gpu_msm2 = get_pool(N);
		auto msm_th1 = std::thread(do_msm_yrrid<0>, vector<void*>{gpu_comb_aw1}, msm_call++, gpu_msm1, N, execSt1);
		auto msm_th2 = std::thread(do_msm_yrrid<1>, vector<void*>{gpu_comb_sw1}, msm_call++, gpu_msm2, N, execSt2);
		
		msm_th1.join();
        msm_th2.join();
		SAFE_CALL(cudaDeviceSynchronize());
        
		release_pool(gpu_msm1, N);
		release_pool(gpu_msm2, N);
#endif

        prover_gpu.perm.release_data(prover_gpu.N, 0);
        /*for (auto& elem : minCounts)
            printf("min counts %d => %d\n", elem.first, elem.second);*/
        
        release_pool(gpu_comb_aw, N);
        release_pool(gpu_comb_aw1, N);
        
        release_pool(gpu_comb_sw, N);
        release_pool(gpu_comb_sw1, N);
        
        release_pool(gpu_powers1, N);
        release_pool(gpu_powers2, N);
        
        div = 1;
        for (int z = 0; z < deep + 1; ++z) {
            release_pool(gpu_tmp1[z], nextN / div);
            release_pool(gpu_tmp2[z], nextN / div);
            
            div *= 2;
        }
    }        
}

#if MSM_YRRID
template<int T>
static void do_msm_yrrid(const std::vector<void*> &scalars_v, int call_num, void* tmp_pool, int N, cudaStream_t st) {
    for (auto& scalars : scalars_v) {
        affine* results = ((affine*)msm_context.result_affine) + call_num;
        void* result_tmp = msm_context.result_tmp[call_num];
            
        coumpute_unmont_kernel((storage*)tmp_pool, (storage*)scalars, N, st);
        SAFE_CALL(cudaStreamSynchronize(st));
        process(MSMContextYrrid[T], results, tmp_pool, N, result_tmp, st);
        //printElem("yrrid", results, 1);
            
        call_num++;
    }	
}
#else
static void do_msm(const std::vector<void*> &scalars_v, int call_num, cudaStream_t st, bool get_memory)
{
    for (auto& scalars : scalars_v) {
        const int windows_count = (BLS_381_FR_MODULUS_BITS - 1) / msm_context.window_bits_count + 1;
        const int precompute_windows_stride = (windows_count - 1) / msm_context.precompute_factor + 1;
        //const int result_bits_count = msm_context.window_bits_count * precompute_windows_stride;
        
        MsmConfiguration cfg;
        
        cfg.bases = msm_context.bases;
        cfg.scalars = scalars; 
        cfg.results = ((point_jacobian*)msm_context.result) + call_num;
        
        cfg.results_tmp = msm_context.result_tmp[call_num];
        if (!get_memory) {
            //printf("coumpute msm %d\n", call_num);
            call_num++;
        }        

        cfg.log_scalars_count = msm_context.log_count;
        cfg.force_min_chunk_size = false;
        cfg.log_min_chunk_size = 0;
        cfg.force_max_chunk_size = true;
        cfg.log_max_chunk_size = msm_context.log_count;
        cfg.window_bits_count = msm_context.window_bits_count;
        cfg.precomputed_windows_stride = precompute_windows_stride;
        cfg.precomputed_bases_stride = 1u << msm_context.log_count;
        cfg.scalars_not_montgomery = false;
        cfg.st = st;
                
        execute_msm(cfg, get_memory);
    }
}
#endif

extern "C" void init_bases(const void* bases, const int count)
{
#if MSM_YRRID
    if (MSMContextYrrid[0] == nullptr) {
        MSMContextYrrid[0] = createContext(381, 1, count);
        preprocessOnlyPoints(MSMContextYrrid[0], (void*)bases, count, nullptr);
    }

    if (MSMContextYrrid[1] == nullptr) {
        MSMContextYrrid[1] = createContext(381, 1, count);
        preprocessOnlyPoints(MSMContextYrrid[1], (void*)bases, count, MSMContextYrrid[0]);
    }    

    if (msm_context.result_affine == nullptr) {
        SAFE_CALL(cudaMallocHost((void**)&msm_context.result_affine, sizeof(affine) * COMMITMENTS));
        for (int z = 0; z < COMMITMENTS; ++z)
            SAFE_CALL(cudaMallocHost((void**)&msm_context.result_tmp[z], sizeof(point_xyzz) * 4096));
    }
#else
    //TODO: allocations
    exit(-1);

    const int WINDOW_BITS_COUNT = 17;
    const int PRECOMPUTE_FACTOR = 8;
    const int WINDOWS_COUNT = (BLS_381_FR_MODULUS_BITS - 1) / WINDOW_BITS_COUNT + 1;
    const int PRECOMPUTE_WINDOWS_STRIDE = (WINDOWS_COUNT - 1) / PRECOMPUTE_FACTOR + 1;
    const int result_bits_count = WINDOW_BITS_COUNT * PRECOMPUTE_WINDOWS_STRIDE;
    
    //printf("msm context: %d %d %d\n", WINDOWS_COUNT, PRECOMPUTE_WINDOWS_STRIDE, result_bits_count);
    const int precompute_factor = PRECOMPUTE_FACTOR;

    const int log_count = std::log2(count);
    
    affine* bases_device = nullptr;
    SAFE_CALL(cudaMalloc((void**)&bases_device, sizeof(affine) * count * precompute_factor));
    //printf("  ==> needed to allocate msm bases for 15H %.2f GB\n", (sizeof(affine) * (1 << (15+7)) * precompute_factor) / 1024. / 1024. / 1024.);
    
    copy_to_device(bases_device, (const affine*)bases, count);

    for (int i = 1; i < precompute_factor; ++i) {
        affine* dst_offset = bases_device + i * count;
        affine* src_offset = bases_device + (i - 1) * count;

        duplicate(dst_offset, src_offset, count);
        
        bls12_381_left_shift_kernel(dst_offset, result_bits_count, count);
        SAFE_CALL(cudaDeviceSynchronize());
    }
    
    msm_context.log_count = log_count;
    msm_context.bases = bases_device;
    msm_context.window_bits_count = WINDOW_BITS_COUNT;
    msm_context.precompute_factor = PRECOMPUTE_FACTOR;
    
    for (int z = 0; z < COMMITMENTS; ++z)
        SAFE_CALL(cudaMallocHost((void**)&msm_context.result_tmp[z], sizeof(point_xyzz) * PRECOMPUTE_FACTOR * WINDOW_BITS_COUNT));
    SAFE_CALL(cudaMallocHost((void**)&msm_context.result, sizeof(point_jacobian) * COMMITMENTS));
    
    do_msm(vector<void*>{nullptr}, 0, execSt1, true);
    if (execSt1 != execSt2)
        do_msm(vector<void*>{nullptr}, 0, execSt2, true);
#endif  
}

extern "C" void init_data(const int count)
{
    msm_call = 0;
    setenv("CUDA_MODULE_LOADING", "EAGER", false);
      
    bool first_alloc = (pi_from_gpu == nullptr);
    pi.clear();
    if (first_alloc) {
        execSt1 = execSt2 = execSt3 = execSt4 = execSt5 = 0;
        /*SAFE_CALL(cudaStreamCreate(&execSt1));
        SAFE_CALL(cudaStreamCreate(&execSt2));
        SAFE_CALL(cudaStreamCreate(&execSt3));
        SAFE_CALL(cudaStreamCreate(&execSt4));
        SAFE_CALL(cudaStreamCreate(&execSt5));*/
    
        //printf("FIRST ALLOC!!\n");
        SAFE_CALL(cudaMallocHost((void**)&pi_from_gpu, sizeof(storage) * count));	        
   
        const int count_data_N = 31-11;
        const int count_data_N8 = 22-12;
        // pre allocate memory
        vector<storage*> p;
        for (int z = 0; z < count_data_N; ++z)
            p.push_back(get_pool(count, true));
        
        for (int z = 0; z < p.size(); ++z)
            release_pool(p[z], count);
        p.clear();
        
        for (int z = 0; z < count_data_N8; ++z)
            p.push_back(get_pool(count * 8, true));

        for (int z = 0; z < p.size(); ++z)
            release_pool(p[z], count * 8);
        p.clear();
        
        int spawn, spawn8, chunk_size, bucket_num;
        setSizesForInverse(spawn, chunk_size, bucket_num, count);
        auto tmp = get_pool(3 * spawn + 1, true);
        release_pool(tmp, 3 * spawn + 1);
        
        setSizesForInverse(spawn8, chunk_size, bucket_num, count * 8);
        if (spawn != spawn8) {
            tmp = get_pool(3 * spawn8 + 1, true);
            release_pool(tmp, 3 * spawn8 + 1);
        }

        alloc_bufs_for_scan(count);
        alloc_bufs_for_scan(8 * count);
        alloc_bufs_for_scan(spawn);
        if (spawn != spawn8)
            alloc_bufs_for_scan(spawn8);
        alloc_bufs_for_scan(128); // for pq arrays

        storage* tmp1 = get_pool(count, true); 
        storage* tmp2 = get_pool(count, true); 
        storage res;
        evaluate_kernel(tmp1, tmp2, res, count, execSt1);
        evaluate_kernel(tmp1, tmp2, res, count, execSt2);
        evaluate_kernel(tmp1, tmp2, res, count, execSt3);

        SAFE_CALL(cudaDeviceSynchronize());
        
        release_pool(tmp1, count);
        release_pool(tmp2, count);
        
        {
            int chunk_size, buckets_num, chunk_size1, deep;
            setSizesForDivide(chunk_size, chunk_size1, buckets_num, deep, count);
            const int nextN = count / chunk_size + 1;
            
            std::vector<storage*> gpu_tmp1;
            std::vector<storage*> gpu_tmp2;
            int div = 1;
            for (int z = 0; z < deep + 1; ++z) {
                gpu_tmp1.push_back(get_pool(nextN / div, true));
                gpu_tmp2.push_back(get_pool(nextN / div, true));
                
                div *= 2;
            }
            
            div = 1;
            for (int z = 0; z < deep + 1; ++z) {
                release_pool(gpu_tmp1[z], nextN / div);
                release_pool(gpu_tmp2[z], nextN / div);
                
                div *= 2;
            }
        }
        
        for (auto& elem : counts)
            minCounts[elem.first] = elem.second;
        
        /*for (auto& elem : pools) {
            printf("%d -> %d\n", elem.first, elem.second.size());
        }*/
    }
    else {
        const int N = count;
        const int domN = count * 8;
        //prover_gpu.check();
        
        if (gpu_w_l_sc != nullptr) release_pool(gpu_w_l_sc, N);// printf("not %d\n", __LINE__);
        if (gpu_w_r_sc != nullptr) release_pool(gpu_w_r_sc, N); //printf("not %d\n", __LINE__);
        if (gpu_w_o_sc  != nullptr) release_pool(gpu_w_o_sc, N);//printf("not %d\n", __LINE__);
        if (gpu_w_4_sc  != nullptr) release_pool(gpu_w_4_sc, N);//printf("not %d\n", __LINE__);

        if (gpu_w_l_poly != nullptr) release_pool(gpu_w_l_poly, N);//printf("not %d\n", __LINE__);
        if (gpu_w_r_poly != nullptr) release_pool(gpu_w_r_poly, N);//printf("not %d\n", __LINE__);
        if (gpu_w_o_poly != nullptr) release_pool(gpu_w_o_poly, N);//printf("not %d\n", __LINE__);
        if (gpu_w_4_poly != nullptr) release_pool(gpu_w_4_poly, N);//printf("not %d\n", __LINE__);

        if (gpu_w_l_poly_ex != nullptr) printf("not %d\n", __LINE__);
        if (gpu_w_r_poly_ex != nullptr) printf("not %d\n", __LINE__);
        if (gpu_w_o_poly_ex != nullptr) printf("not %d\n", __LINE__);
        if (gpu_w_4_poly_ex != nullptr) printf("not %d\n", __LINE__);

        if (gpu_omegas != nullptr) printf("not %d\n", __LINE__);
        if (gpu_omegas_8n != nullptr) printf("not %d\n", __LINE__);
        if (gpu_omegas_inv != nullptr) printf("not %d\n", __LINE__);
        if (gpu_omegas_inv8n != nullptr) printf("not %d\n", __LINE__);

        if (gpu_pq != nullptr) printf("not %d\n", __LINE__);
        if (gpu_pq_inv != nullptr) printf("not %d\n", __LINE__);
        if (gpu_pq_8n != nullptr) printf("not %d\n", __LINE__);
        if (gpu_pq_inv8n != nullptr) printf("not %d\n", __LINE__);

        if (gpu_powers != nullptr) printf("not %d\n", __LINE__);
        if (gpu_quotient != nullptr) release_pool(gpu_quotient, domN);//printf("not %d\n", __LINE__);
        if (gpu_linear != nullptr) release_pool(gpu_linear, N);//printf("not %d\n", __LINE__);

        if (gpu_pi_poly != nullptr) release_pool(gpu_pi_poly, N);//printf("not %d\n", __LINE__);
        if (gpu_pi_poly_ex != nullptr) release_pool(gpu_pi_poly_ex, domN);//printf("not %d\n", __LINE__);

        //if (pi_from_gpu != nullptr) printf("not %d\n", __LINE__);
        if (gpu_z_poly != nullptr) release_pool(gpu_z_poly, N);//printf("not %d\n", __LINE__);   
        for (auto& elem : pools) {
            if (elem.second.size() != counts[elem.first]) {
                //printf("%d -> %d %d\n", elem.first, elem.second.size(), counts[elem.first]);
                
                while (elem.second.size())
                    elem.second.pop();
                
                for (auto& pp : created[elem.first])
                    elem.second.push(pp);
                
                //printf("%d -> %d %d\n", elem.first, elem.second.size(), counts[elem.first]);
            }
        }
    }
    
    debugPrint = true;
}
