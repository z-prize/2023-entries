#pragma once

#include "ff_config.cuh"
#include "ff_storage.cuh"
#include "ff_dispatch_st.cuh"
#include "memory.cuh"

typedef fd_q::storage storage;

void alloc_bufs_for_scan(const int N);
void poly_compute_scan(storage* poly_out, const storage* poly_in, const int N, cudaStream_t st = 0);
void get_powers_kernel(storage* powers, const storage& g, const int N, cudaStream_t st = 0);
void distribute_powers_kernel(storage* poly, const storage* powers, const int N, cudaStream_t st = 0);
void scale(storage* gpu_poly, const storage& scale, const int N, cudaStream_t st = 0);
void set_value(storage* poly, const storage& value, const int elem = 0, cudaStream_t st = 0);
void compute_gate_constraint_kernel(storage* gate_constraint,
                                    const storage* wl,
                                    const storage* wr,
                                    const storage* wo,
                                    const storage* w4,
                                    
                                    const storage* q_m,
                                    const storage* q_l,
                                    const storage* q_r,
                                    const storage* q_o,
                                    const storage* q_4,
                                    const storage* q_hl,
                                    const storage* q_hr,
                                    const storage* q_h4,
                                    const storage* q_c,
                                    const storage* q_arith,

                                    const storage* pi_eval,
                                    const int N, cudaStream_t st = 0);
void compute_permutation_kernel(storage* permutation,
                                const storage* wl,
                                const storage* wr,
                                const storage* wo,
                                const storage* w4,
                                const storage* z,
                                const storage* linear,
                                const storage* l1_alpha,

                                const storage* left_sig,
                                const storage* right_sig,
                                const storage* out_sig,
                                const storage* fourth_sig,

                                const storage alpha, const storage beta, const storage gamma,
                                const storage K1, const storage K2, const storage K3,
                                const int N, cudaStream_t st = 0);
void compute_quotient_kernel(storage* poly, const storage* gate_constraints, const storage* permutation,
                             const storage* denominator, const int N, cudaStream_t st = 0);
int get_invesion_blocks_count();                             
void invesion_kernel(storage* poly, storage* result, storage* buckets, const int spawn,
                     const int gws, const int lws, const int chunk_size, const int bucket_num, const int N,
                     cudaStream_t st = 0);
void compute_z_poly_denum_kernel(storage* result, const storage* wires_0, const storage* wires_1, const storage* wires_2,
                                 const storage* wires_3, const storage* sigma_0, const storage* sigma_1, const storage* sigma_2,
                                 const storage* sigma_3, const storage& beta, const storage& gamma, const int N, cudaStream_t st = 0);
void compute_z_poly_kernel(storage* result, const storage* denum_inv,
                           const storage* wires_0, const storage* wires_1, const storage* wires_2, const storage* wires_3,
                           const storage* roots, const storage& K1, const storage& K2, const storage& K3,
                           const storage& beta, const storage& gamma, const int N, cudaStream_t st);
void evaluate_kernel(const storage* poly, const storage* powers, storage& res, const int N, cudaStream_t st = 0);
void compress_quotient_term_kernel(storage* poly, const storage* q, const storage& vanish, const storage& zeta, const int N, cudaStream_t st = 0);
void compute_linearisation_kernel(storage* poly,
                                  storage* buf,
                                  const storage* z_poly,
                                  const storage* prov_4,

                                  const storage& zeta, const storage& alpha, const storage& beta, const storage& gamma,
                                  const storage& a_eval, const storage& b_eval, const storage& c_eval, const storage& d_eval,
                                  const storage& sigma_1_eval, const storage& sigma_2_eval, const storage& sigma_3_eval,
                                  const storage& z_eval, const storage& l1_alpha_sq, const int N, cudaStream_t st = 0);
void compute_linearisation_kernel(storage *poly, const storage *perm, const storage *qterm, 
                                  const storage *q_m, const storage *q_l, const storage *q_r, const storage *q_o, 
                                  const storage *q_4, const storage *q_hl, const storage *q_hr, const storage *q_h4, const storage *q_c,
                                  const storage& a_val, const storage& b_val, const storage& c_val, const storage& d_val, const storage& q_arith_val,
                                  const int N, cudaStream_t st = 0);
void coumpute_unmont_kernel(storage* poly, const storage* from, const int N, cudaStream_t st = 0);
void compute_pq_kernel(storage *poly, const storage& omega, const int max_deg, const int N, cudaStream_t st = 0);
void compute_combination_aw_kernel(storage* poly, 
                                   const storage* lin_poly, 
                                   const storage* prov_sl,
                                   const storage* prov_sr,
                                   const storage* prov_so,
                                   const storage* w_l_poly,
                                   const storage* w_r_poly,
                                   const storage* w_o_poly,
                                   const storage* w_4_poly,
                                   const storage& zeta,
                                   const int N, cudaStream_t st = 0);
void compute_combination_ws_kernel(storage *poly, 
                                   const storage* z_poly, 
                                   const storage* w_l_poly,
                                   const storage* w_r_poly,
                                   const storage* w_4_poly,
                                   const storage& zeta,
                                   const int N, cudaStream_t st = 0);
void compute_poly_division_kernel(storage *poly, storage *poly1, storage **result, storage* powers, const storage& p, 
                                  int buckets_num, int chunk_size, int deep, int chunk_size1, int N, cudaStream_t st = 0);
void compute_fft_kernel(storage* gpu_poly, storage* tmp_buffer, const storage* pq_buffer, const storage *omegas_buffer, const int n,
                        const int lgp, const int deg, const int max_deg, const int gws, const int lws, cudaStream_t st = 0);