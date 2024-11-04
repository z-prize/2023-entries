#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <map>
#include <vector>
#include <cassert>

#include <cub/cub.cuh>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/vector.h>

#include "poly_functions.h"
#include "cached_allocator.h"
#include "functors.h"

#define GET_GLOBAL_ID() blockIdx.x * blockDim.x + threadIdx.x

static std::map<int, std::pair<storage*, size_t>> bufs;
static std::map<cudaStream_t, cached_allocator> alloc;

static int get_bl(const int count, const int th)
{
    return (count / th) + ((count % th) != 0);
}

void alloc_bufs_for_scan(const int N)
{
    if (bufs.count(N) == 0) {
        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, (const storage*)nullptr, (storage*)nullptr, functor_mul(), N);

        // Allocate temporary storage for inclusive prefix scan
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        bufs[N] = std::make_pair((storage*)d_temp_storage, temp_storage_bytes);
        
        //printf("allocate for scan %llu\n", temp_storage_bytes);
    }
}

void poly_compute_scan(storage* poly_out, const storage* poly_in, const int N, cudaStream_t st)
{
    auto it = bufs.find(N);
    if (it == bufs.end()) {
        printf("buf for %d elems for scan not found\n", N);
        exit(-1);
    }
    cub::DeviceScan::InclusiveScan(it->second.first, it->second.second, poly_in, poly_out, functor_mul(), N, st);
}

__global__ void poly_set_constant(storage *poly, const storage g, uint n) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;

    if (gid == 0)
        memory::store(poly + gid, fd_q::get_one());
    else
        memory::store(poly + gid, g);
}

void get_powers_kernel(storage* powers, const storage& g, const int N, cudaStream_t st) {
    poly_set_constant<<<get_bl(N, 64), 64, 0, st>>>(powers, g, N);
    poly_compute_scan(powers, powers, N, st);
}

__global__ void poly_mul_assign(storage *poly_1, const storage *poly_2, uint n) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;

    memory::store(poly_1 + gid, fd_q::mul(memory::load(poly_1 + gid), memory::load(poly_2 + gid)));
}

void distribute_powers_kernel(storage* poly, const storage* powers, const int N, cudaStream_t st)
{
    poly_mul_assign<<<get_bl(N, 64), 64, 0, st>>>(poly, powers, N);
}

__global__ void poly_scale(storage *poly, const storage scale, uint n) {
    const uint gid = GET_GLOBAL_ID();

    if (gid >= n) return;
    memory::store(poly + gid, fd_q::mul(memory::load(poly + gid), scale));
}

void scale(storage* gpu_poly, const storage& scale, const int N, cudaStream_t st)
{
    poly_scale<<<get_bl(N, 64), 64, 0, st>>> (gpu_poly, scale, N);
}

__global__ void poly_set_value(storage* poly, const storage value, const int elem)
{
    poly[elem] = value;
}

void set_value(storage* poly, const storage& value, const int elem, cudaStream_t st)
{
    poly_set_value<<<1, 1, 0, st>>>(poly, value, elem);
}

__global__ void poly_compute_gate_constraint(storage* gate_constraint,
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

                                             const storage* pi,
                                             uint n)
{
    uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;

    storage a_val = memory::load(wl + gid);
    storage b_val = memory::load(wr + gid);
    storage c_val = memory::load(wo + gid);
    storage d_val = memory::load(w4 + gid);
    
    storage arithmetic = (fd_q::mul(a_val , memory::load(q_l + gid)));
    arithmetic = fd_q::add(arithmetic, (fd_q::mul(b_val, memory::load(q_r + gid))));
    arithmetic = fd_q::add(arithmetic, (fd_q::mul(c_val, memory::load(q_o + gid))));
    arithmetic = fd_q::add(arithmetic, (fd_q::mul(d_val, memory::load(q_4 + gid))));
    arithmetic = fd_q::add(arithmetic, (fd_q::mul(fd_q::pow5(a_val), memory::load(q_hl + gid))));
    arithmetic = fd_q::add(arithmetic, (fd_q::mul(fd_q::pow5(b_val), memory::load(q_hr + gid))));
    arithmetic = fd_q::add(arithmetic, (fd_q::mul(fd_q::pow5(d_val), memory::load(q_h4 + gid))));
    arithmetic = fd_q::add(arithmetic, memory::load(q_c + gid));
    arithmetic = fd_q::mul(arithmetic, memory::load(q_arith + gid));


    memory::store(gate_constraint + gid, fd_q::add(arithmetic, memory::load(pi + gid)));
}

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
                                    const int N, cudaStream_t st)
{
    poly_compute_gate_constraint<<< get_bl(N, 64), 64, 0, st>>> (gate_constraint, wl, wr, wo, w4, q_m, q_l, q_r, q_o, q_4, q_hl, q_hr, q_h4, q_c, q_arith, pi_eval, N);
}

__global__ void poly_compute_permutation(storage* permutation,
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
                                         uint n)
{
    uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;

    storage x = memory::load(linear + gid);

    storage a = fd_q::add(fd_q::add(memory::load(wl + gid), fd_q::mul(beta, x)), gamma);
    a = fd_q::mul(a, fd_q::add(fd_q::add(memory::load(wr + gid), fd_q::mul(K1, x)), gamma));
    a = fd_q::mul(a, fd_q::add(fd_q::add(memory::load(wo + gid), fd_q::mul(K2, x)), gamma));
    a = fd_q::mul(a, fd_q::add(fd_q::add(memory::load(w4 + gid), fd_q::mul(K3, x)), gamma));
    a = fd_q::mul(a, memory::load(z + gid));
    a = fd_q::mul(a, alpha);

    storage b = fd_q::add(fd_q::add(memory::load(wl + gid), fd_q::mul(beta, memory::load(left_sig + gid))), gamma);
    b = fd_q::mul(b, fd_q::add(fd_q::add(memory::load(wr + gid), fd_q::mul(beta, memory::load(right_sig + gid))), gamma));
    b = fd_q::mul(b, fd_q::add(fd_q::add(memory::load(wo + gid), fd_q::mul(beta, memory::load(out_sig + gid))), gamma));
    b = fd_q::mul(b, fd_q::add(fd_q::add(memory::load(w4 + gid), fd_q::mul(beta, memory::load(fourth_sig + gid))), gamma));
    b = fd_q::mul(b, memory::load(z + ((gid + 8) % n)));
    b = fd_q::mul(b, alpha);
    b = fd_q::neg(b);
    
    storage c = fd_q::mul(fd_q::sub(memory::load(z + gid), fd_q::get_one()), memory::load(l1_alpha + gid));

    memory::store(permutation + gid, fd_q::add(a, fd_q::add(b, c)));
}

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
                                const int N, cudaStream_t st)
{
    poly_compute_permutation<<< get_bl(N, 64), 64, 0, st>>> (permutation, wl, wr, wo, w4, z, linear, l1_alpha, left_sig, right_sig, out_sig, fourth_sig, alpha, beta, gamma, K1, K2, K3, N);
}


__global__ void poly_compute_quotient(storage* poly,
                                      const storage* gate_constraints,
                                      const storage* permutation,
                                      const storage* denominator,
                                      const int n)
{
    uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;

    memory::store(poly + gid, fd_q::mul(fd_q::add(memory::load(gate_constraints + gid), memory::load(permutation + gid)), memory::load(denominator + gid)));
}

void compute_quotient_kernel(storage* poly, const storage* gate_constraints, const storage* permutation,
                             const storage* denominator, const int N, cudaStream_t st)
{
    poly_compute_quotient<<<get_bl(N, 64), 64, 0, st>>> (poly, gate_constraints, permutation, denominator, N);
}

__global__ void poly_batch_inversion_part_1(const storage *poly_1,
                                            storage *result,
                                            storage *result_inv,
                                            storage *buckets,
                                            uint buckets_num,
                                            uint chunk_size,
                                            int spawn,
                                            uint n) {
    int gid = GET_GLOBAL_ID();

    if (gid * chunk_size >= n) {
      memory::store(result + gid, fd_q::get_one());
      if (spawn - gid - 1 >= 0)
          memory::store(result_inv + spawn - gid - 1, fd_q::get_one());
      return;
    }

    // offset for gid
    buckets = buckets + gid * buckets_num;
    uint bucket_size = (uint)ceil(chunk_size/(float)buckets_num);

    uint start = gid * chunk_size;
    uint end = min(start + chunk_size, n);
    uint bucket_idx = 0;

    storage acc = fd_q::get_one();

    for (uint i=start; i<end; i+=bucket_size) {
    uint idx_end = min(i+bucket_size, end);

    for (uint j=i; j<idx_end; j++) {
      acc = fd_q::mul(acc, memory::load(poly_1 + j));
    }

    memory::store(buckets + bucket_idx, acc);
    bucket_idx++;
    }

    memory::store(result + gid, acc);
    memory::store(result_inv + spawn - gid - 1, acc);
}

__global__ void poly_batch_inversion_part_2(storage *poly_1,
                                            const storage *result,
                                            const storage *buckets,
                                            uint buckets_num,
                                            uint chunk_size,
                                            uint n) {
    uint gid = GET_GLOBAL_ID();

    if (gid * chunk_size >= n) return;

    uint start = gid * chunk_size;
    uint end = min(start + chunk_size, n);

    // offset for gid
    buckets = buckets + gid * buckets_num;
    uint bucket_size = (uint)ceil(chunk_size/(float)buckets_num);

    storage acc = memory::load(result + gid);
    storage bucket_acc = fd_q::get_one();

    for (uint i=(end - 1); i>start; i--) {
        // get bucket index
        uint bucket_idx = (uint)((i - start)/(float)bucket_size);

        // get previous bucket
        if (bucket_idx > 0) {
          bucket_acc = memory::load(buckets + bucket_idx-1);
        } else {
          bucket_acc = fd_q::get_one();
        }

        // acc_i = (previous bucket) * (up to i elements in this bucket)
        for (uint j=bucket_idx * bucket_size + gid * chunk_size; j<i; j++)
          bucket_acc = fd_q::mul(bucket_acc, memory::load(poly_1 + j));

        storage tmp = memory::load(poly_1 + i);
        memory::store(poly_1 + i, fd_q::mul(acc, bucket_acc));
        acc = fd_q::mul(acc, tmp);
    }

    memory::store(poly_1 + start, acc);
}

__global__ void poly_inverse_elem(storage *to_inv, storage *out)
{
    uint gid = GET_GLOBAL_ID();
    if (gid > 0) return;
    
    out[0] = fd_q::inverse(to_inv[0]);
}

__global__ void poly_inverse_spawn(storage *result, const storage *res_mul, const storage *res_rev_mul, const storage *inv, const int n)
{
    uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    
    const storage acc = memory::load(inv + 0);
    
    if (gid == (n - 1))
        memory::store(result + gid, fd_q::mul(memory::load(res_mul + gid - 1), acc));
    else if (gid == 0)
        memory::store(result + gid, fd_q::mul(memory::load(res_rev_mul + n - (gid + 1) - 1), acc));
    else
        memory::store(result + gid, fd_q::mul(fd_q::mul(memory::load(res_mul + gid - 1), acc), memory::load(res_rev_mul + n - (gid + 1) - 1)));
}

int get_invesion_blocks_count()
{
    int blocks1 = 0;
    int blocks2 = 0;
    int threads = 32;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks1, (void*)poly_batch_inversion_part_1, threads, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks2, (void*)poly_batch_inversion_part_2, threads, 0);
    
    return std::min(blocks1, blocks2);
}

void invesion_kernel(storage* poly, storage* result, storage* buckets, const int spawn,
                     const int gws, const int lws, const int chunk_size, const int bucket_num, const int N,
                     cudaStream_t st)
{
    storage* result_inv = result + spawn;
    storage* result_mul = result + 2 * spawn;
    storage* inv = result + 3 * spawn;

    poly_batch_inversion_part_1 <<<gws, lws, 0, st>>> (poly, result, result_inv, buckets, bucket_num, chunk_size, spawn, N);
        
    poly_compute_scan(result_mul, result, spawn, st);
    poly_compute_scan(result_inv, result_inv, spawn, st);

    poly_inverse_elem<<<1, 1, 0, st>>>(result_mul + spawn - 1, inv);        
    poly_inverse_spawn<<<get_bl(spawn, 32), 32, 0, st>>>(result, result_mul, result_inv, inv, spawn);

    poly_batch_inversion_part_2 <<<gws, lws, 0, st >>> (poly, result, buckets, bucket_num, chunk_size, N);
}

__global__ void poly_compute_z_denum(storage* result,
                                     const storage* wires_0,
                                     const storage* wires_1,
                                     const storage* wires_2,
                                     const storage* wires_3,
                                     const storage* sigma_0,
                                     const storage* sigma_1,
                                     const storage* sigma_2,
                                     const storage* sigma_3,
                                     const storage beta, const storage gamma,
                                     const int n)
{
    uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;

    //w + beta * sigma + gamma
    storage tmp = (fd_q::add(fd_q::add(memory::load(wires_0 + gid), fd_q::mul(memory::load(sigma_0 + gid), beta)), gamma));
    tmp = fd_q::mul(tmp, (fd_q::add(fd_q::add(memory::load(wires_1 + gid), fd_q::mul(memory::load(sigma_1 + gid), beta)), gamma)));
    tmp = fd_q::mul(tmp, (fd_q::add(fd_q::add(memory::load(wires_2 + gid), fd_q::mul(memory::load(sigma_2 + gid), beta)), gamma)));
    tmp = fd_q::mul(tmp, (fd_q::add(fd_q::add(memory::load(wires_3 + gid), fd_q::mul(memory::load(sigma_3 + gid), beta)), gamma)));

    memory::store(result + gid, tmp);
}

void compute_z_poly_denum_kernel(storage* result, const storage* wires_0, const storage* wires_1, const storage* wires_2,
                                 const storage* wires_3, const storage* sigma_0, const storage* sigma_1, const storage* sigma_2,
                                 const storage* sigma_3, const storage& beta, const storage& gamma, const int N, cudaStream_t st)
{
    poly_compute_z_denum<<<get_bl(N, 64), 64, 0, st>>>(result, wires_0, wires_1, wires_2, wires_3, sigma_0, sigma_1, sigma_2, sigma_3, beta, gamma, N);
}

__global__ void poly_compute_z(storage* result,
                               const storage* denum_inv,
                               const storage* wires_0,
                               const storage* wires_1,
                               const storage* wires_2,
                               const storage* wires_3,
                               const storage* roots,
                               const storage K1, const storage K2, const storage K3,
                               const storage beta, const storage gamma,
                               const int n)
{
    uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    
    if (gid == 0) {
        memory::store(result + gid, fd_q::get_one());
        return;
    }
    gid--;
    
    const auto root = memory::load(roots + gid);    
    //w + k_beta * root + gamma
    storage tmp = (fd_q::add(fd_q::add(memory::load(wires_0 + gid), fd_q::mul(beta, root)), gamma));
    tmp = fd_q::mul(tmp, (fd_q::add(fd_q::add(memory::load(wires_1 + gid), fd_q::mul(K1, root)), gamma)));
    tmp = fd_q::mul(tmp, (fd_q::add(fd_q::add(memory::load(wires_2 + gid), fd_q::mul(K2, root)), gamma)));
    tmp = fd_q::mul(tmp, (fd_q::add(fd_q::add(memory::load(wires_3 + gid), fd_q::mul(K3, root)), gamma)));

    memory::store(result + gid + 1, fd_q::mul(tmp, memory::load(denum_inv + gid)));
}

void compute_z_poly_kernel(storage* result, const storage* denum_inv,
                           const storage* wires_0, const storage* wires_1, const storage* wires_2, const storage* wires_3,
                           const storage* roots, const storage& K1, const storage& K2, const storage& K3,
                           const storage& beta, const storage& gamma, const int N, cudaStream_t st)
{
    poly_compute_z <<<get_bl(N, 64), 64, 0, st>>>(result, denum_inv, wires_0, wires_1, wires_2, wires_3, roots, K1, K2, K3, beta, gamma, N);
    poly_compute_scan(result, result, N, st);
}

void evaluate_kernel(const storage* poly, const storage* powers, storage& res, const int N, cudaStream_t st)
{
    auto exec = thrust::cuda::par(alloc[st]).on(st);

    res = thrust::transform_reduce(exec,
                                   thrust::make_zip_iterator(poly, powers),
                                   thrust::make_zip_iterator(poly + N, powers + N),
                                   functor_mul_2(),
                                   fd_q::get_zero(),
                                   functor_add());
}

__global__ void poly_compress_quotient_term(storage* poly,
                                            const storage* q,
                                            const storage vanish,
                                            const storage zeta,
                                            const int n)
{

    uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;

    const storage* q1 = q;
    const storage* q2 = q + 1 * n;
    const storage* q3 = q + 2 * n;
    const storage* q4 = q + 3 * n;
    const storage* q5 = q + 4 * n;
    const storage* q6 = q + 5 * n;

    //((((((((((t_6_poly * z) + t_5_poly) * z) + t_4_poly) * z) + t_3_poly) * z) + t_2_poly) * z)+ t_1_poly) * vanish;

    storage tmp = fd_q::add(fd_q::mul(memory::load(q6 + gid), zeta), memory::load(q5 + gid));
    tmp = fd_q::add(fd_q::mul(tmp, zeta), memory::load(q4 + gid));
    tmp = fd_q::add(fd_q::mul(tmp, zeta), memory::load(q3 + gid));
    tmp = fd_q::add(fd_q::mul(tmp, zeta), memory::load(q2 + gid));
    tmp = fd_q::add(fd_q::mul(tmp, zeta), memory::load(q1 + gid));

    //TODO: replace to constant
    storage minus1 = fd_q::neg(fd_q::get_one());
    memory::store(poly + gid, fd_q::mul(fd_q::mul(tmp, vanish), minus1));
}

void compress_quotient_term_kernel(storage* poly, const storage* q, const storage& vanish, const storage& zeta, const int N, cudaStream_t st)
{
    poly_compress_quotient_term<<<get_bl(N, 64), 64, 0, st>>>(poly, q, vanish, zeta, N);
}

__global__ void poly_compute_abc(storage *buf, storage zeta,
                                 storage a_eval, storage b_eval, storage c_eval, storage d_eval,
                                 storage alpha, storage beta, storage gamma,
                                 storage sigma_1, storage sigma_2, storage sigma_3,
                                 storage z_eval)
{
    uint gid = GET_GLOBAL_ID();

    storage beta_z = fd_q::mul(beta, zeta);

    // a_eval + beta * z_challenge + gamma
    storage a_0 = fd_q::add(a_eval, beta_z);
    a_0 = fd_q::add(a_0, gamma);

    // b_eval + beta * K1 * z_challenge + gamma
    storage beta_z_k1 = fd_q::mul(fd_q::get_k1(), beta_z);
    storage a_1 = fd_q::add(b_eval, beta_z_k1);
    a_1 = fd_q::add(a_1, gamma);

    // c_eval + beta * K2 * z_challenge + gamma
    storage beta_z_k2 = fd_q::mul(fd_q::get_k2(), beta_z);
    storage a_2 = fd_q::add(c_eval, beta_z_k2);
    a_2 = fd_q::add(a_2, gamma);

    // d_eval + beta * K3 * z_challenge + gamma
    storage beta_z_k3 = fd_q::mul(fd_q::get_k3(), beta_z);
    storage a_3 = fd_q::add(d_eval, beta_z_k3);
    a_3 = fd_q::add(a_3, gamma);

    storage a = fd_q::mul(a_0, a_1);
    a = fd_q::mul(a, a_2);
    a = fd_q::mul(a, a_3);
    a = fd_q::mul(a, alpha);

    if (gid == 0)
        memory::store(buf + 0, a);

    storage beta_sigma_1 = fd_q::mul(beta, sigma_1);
    a_0 = fd_q::add(a_eval, beta_sigma_1);
    a_0 = fd_q::add(a_0, gamma);

    // b_eval + beta * sigma_2 + gamma
    storage beta_sigma_2 = fd_q::mul(beta, sigma_2);
    a_1 = fd_q::add(b_eval, beta_sigma_2);
    a_1 = fd_q::add(a_1, gamma);

    // c_eval + beta * sigma_3 + gamma
    storage beta_sigma_3 = fd_q::mul(beta, sigma_3);
    a_2 = fd_q::add(c_eval, beta_sigma_3);
    a_2 = fd_q::add(a_2, gamma);

    storage beta_z_eval = fd_q::mul(beta, z_eval);

    a = fd_q::mul(fd_q::mul(a_0, a_1), a_2);
    a = fd_q::mul(a, beta_z_eval);
    a = fd_q::mul(a, alpha);
    a = fd_q::neg(a);

    if (gid == 0)
        memory::store(buf + 1, a);
}

__global__ void poly_compute_linearisation(storage* poly, const storage* buf, const storage* z_poly, const storage* prov_4, storage l1_alpha_sq, const int n)
{
    uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;

    auto z = memory::load(z_poly + gid);
    auto a = fd_q::mul(z, memory::load(buf + 0));
    auto b = fd_q::mul(memory::load(prov_4 + gid), memory::load(buf + 1));
    auto c = fd_q::mul(z, l1_alpha_sq);

    memory::store(poly + gid, fd_q::add(fd_q::add(a, b), c));
}

void compute_linearisation_kernel(storage* poly,
                                  storage* buf,
                                  const storage* z_poly,
                                  const storage* prov_4,

                                  const storage& zeta,
                                  const storage& alpha,
                                  const storage& beta,
                                  const storage& gamma,
                                  const storage& a_eval,
                                  const storage& b_eval,
                                  const storage& c_eval,
                                  const storage& d_eval,
                                  const storage& sigma_1_eval,
                                  const storage& sigma_2_eval,
                                  const storage& sigma_3_eval,
                                  const storage& z_eval,
                                  const storage& l1_alpha_sq,
                                  const int N, cudaStream_t st)
{
    poly_compute_abc<<<1, 1, 0, st>>>(buf, zeta, a_eval, b_eval, c_eval, d_eval, alpha, beta, gamma, sigma_1_eval, sigma_2_eval, sigma_3_eval, z_eval);
    poly_compute_linearisation<<<get_bl(N, 64), 64, 0, st>>>(poly, buf, z_poly, prov_4, l1_alpha_sq, N);
}

__global__ void poly_compute_linearisation(storage *poly,
                                           const storage *perm, 
                                           const storage *qterm,
                                           const storage *q_m,
                                           const storage *q_l,
                                           const storage *q_r,
                                           const storage *q_o,
                                           const storage *q_4,
                                           const storage *q_hl,
                                           const storage *q_hr,
                                           const storage *q_h4,
                                           const storage *q_c,
                                           const storage a_eval,
                                           const storage b_eval,
                                           const storage c_eval,
                                           const storage d_eval,
                                           const storage q_arith_eval,
                                           const int n)
{
    uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;

    /*storage arithmetic = fd_q::mul(a_eval, b_eval);
    arithmetic = fd_q::mul(arithmetic, q_m[gid]);
    storage arithmetic = fd_q::add(arithmetic, fd_q::mul(q_l[gid], a_eval));*/
    
    storage arithmetic = fd_q::mul(memory::load(q_l + gid), a_eval);
    arithmetic = fd_q::add(arithmetic, fd_q::mul(memory::load(q_r + gid), b_eval));
    arithmetic = fd_q::add(arithmetic, fd_q::mul(memory::load(q_o + gid), c_eval));
    arithmetic = fd_q::add(arithmetic, fd_q::mul(memory::load(q_4 + gid), d_eval));
    arithmetic = fd_q::add(arithmetic, fd_q::mul(memory::load(q_hl + gid), fd_q::pow5(a_eval)));
    arithmetic = fd_q::add(arithmetic, fd_q::mul(memory::load(q_hr + gid), fd_q::pow5(b_eval)));
    arithmetic = fd_q::add(arithmetic, fd_q::mul(memory::load(q_h4 + gid), fd_q::pow5(d_eval)));
    arithmetic = fd_q::mul(fd_q::add(arithmetic, memory::load(q_c + gid)), q_arith_eval);

    memory::store(poly + gid, fd_q::add(fd_q::add(arithmetic, memory::load(perm + gid)), memory::load(qterm + gid))); 
}

void compute_linearisation_kernel(storage *poly, const storage *perm, const storage *qterm, 
                                  const storage *q_m, const storage *q_l, const storage *q_r, const storage *q_o, 
                                  const storage *q_4, const storage *q_hl, const storage *q_hr, const storage *q_h4, const storage *q_c,
                                  const storage& a_val, const storage& b_val, const storage& c_val, const storage& d_val, const storage& q_arith_val,
                                  const int N, cudaStream_t st)
{    
    poly_compute_linearisation<<<get_bl(N, 64), 64, 0, st>>>(poly, perm, qterm, q_m, q_l, q_r, q_o, q_4, q_hl, q_hr, q_h4, q_c,a_val, b_val, c_val, d_val, q_arith_val, N);
}

__global__ void poly_unmont(storage *poly, const storage *mont, uint n) {
	uint gid = GET_GLOBAL_ID();
	if (gid >= n) return;

	auto one = fd_q::get_zero();
	one.limbs[0] = 1;
	memory::store(poly + gid, fd_q::mul(memory::load(mont + gid), one));
}

void coumpute_unmont_kernel(storage* poly, const storage* from, const int N, cudaStream_t st)
{
    poly_unmont<<<get_bl(N, 64), 64, 0, st>>>(poly, from, N);
}

__device__ storage poly_pow(storage base, uint exponent) {
    auto res = fd_q::get_one();
    while(exponent > 0) {
        if (exponent & 1)
            res = fd_q::mul(res, base);
        exponent = exponent >> 1;
        base = fd_q::sqr(base);
    }
    return res;
}

__global__ void poly_compute_pq(storage* pq, storage omega, const int max_deg, const int N, const int pq_elems)
{
    uint gid = GET_GLOBAL_ID();
    if (gid >= pq_elems) return;
    
    storage tw = poly_pow(omega, N >> max_deg);
    if (gid == 0)
        pq[gid] = fd_q::get_one();
    else
        pq[gid] = tw;
}

void compute_pq_kernel(storage *poly, const storage& omega, const int max_deg, const int N, cudaStream_t st)
{
    const int elems = 1 << max_deg >> 1;
    poly_compute_pq<<<get_bl(elems, 32), 32, 0, st>>>(poly, omega, max_deg, N, elems);
    poly_compute_scan(poly, poly, elems, st);
}

__global__ void poly_compute_combination_aw(storage *poly, 
                                            const storage* lin_poly, 
                                            const storage* prov_sl,
                                            const storage* prov_sr,
                                            const storage* prov_so,
                                            const storage* w_l_poly,
                                            const storage* w_r_poly,
                                            const storage* w_o_poly,
                                            const storage* w_4_poly,
                                            storage zeta,
                                            const int n)
{
    uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    
    storage sum = fd_q::mul(zeta, memory::load(w_4_poly + gid));
    
    sum = fd_q::add(sum, memory::load(w_o_poly + gid));
    sum = fd_q::mul(sum, zeta);
    
    sum = fd_q::add(sum, memory::load(w_r_poly + gid));
    sum = fd_q::mul(sum, zeta);
    
    sum = fd_q::add(sum, memory::load(w_l_poly + gid));
    sum = fd_q::mul(sum, zeta);
    
    sum = fd_q::mul(sum, zeta); //table
    sum = fd_q::mul(sum, zeta); //h_2
    sum = fd_q::mul(sum, zeta); //f
    
    sum = fd_q::add(sum, memory::load(prov_so + gid));
    sum = fd_q::mul(sum, zeta);
    
    sum = fd_q::add(sum, memory::load(prov_sr + gid));
    sum = fd_q::mul(sum, zeta);
    
    sum = fd_q::add(sum, memory::load(prov_sl + gid));
    sum = fd_q::mul(sum, zeta);

    memory::store(poly + n - 1 - gid, fd_q::add(sum, memory::load(lin_poly + gid)));
}

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
                                   const int N, cudaStream_t st)
{
    poly_compute_combination_aw<<<get_bl(N, 64), 64, 0, st>>>(poly, lin_poly, prov_sl, prov_sr, prov_so, w_l_poly, w_r_poly, w_o_poly, w_4_poly, zeta, N);
}

__global__ void poly_compute_combination_sw(storage *poly, 
                                            const storage* z_poly, 
                                            const storage* w_l_poly,
                                            const storage* w_r_poly,
                                            const storage* w_4_poly,
                                            storage zeta,
                                            const int n)
{
    uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    
    storage z2_poly = (gid == 0) ? fd_q::get_one() : fd_q::get_zero();

    storage sum = fd_q::mul(zeta, z2_poly);
    sum = fd_q::mul(sum, zeta); // h1
    
    sum = fd_q::add(sum, memory::load(w_4_poly + gid));
    sum = fd_q::mul(sum, zeta);
    
    sum = fd_q::add(sum, memory::load(w_r_poly + gid));
    sum = fd_q::mul(sum, zeta);
    
    sum = fd_q::add(sum, memory::load(w_l_poly + gid));
    sum = fd_q::mul(sum, zeta);
    
    memory::store(poly + n - 1 - gid, fd_q::add(sum, memory::load(z_poly + gid)));
}

void compute_combination_ws_kernel(storage *poly, 
                                   const storage* z_poly, 
                                   const storage* w_l_poly,
                                   const storage* w_r_poly,
                                   const storage* w_4_poly,
                                   const storage& zeta,
                                   const int N, cudaStream_t st)
{
    poly_compute_combination_sw<<<get_bl(N, 64), 64, 0, st>>>(poly, z_poly, w_l_poly, w_r_poly, w_4_poly, zeta, N);
}

__global__ void poly_batch_division_part_1(storage *poly,
                                           storage *result,
                                           const storage *powers,
                                           const uint currP,
                                           uint buckets_num,
                                           uint chunk_size,
                                           uint n) {
    uint gid = GET_GLOBAL_ID();

    if (gid >= buckets_num)
        return;
    if (gid * chunk_size >= n)
        return;

    storage p = powers[currP];
    uint start = gid * chunk_size;
    uint end = min(start + chunk_size, n);
    
    storage acc = poly[start];
    for (uint j = start + 1; j < end; ++j) {
        acc = fd_q::add(fd_q::mul(acc, p), poly[j]);
        memory::store(poly + j, acc);
    }

    memory::store(result + gid, acc);
}

__global__ void poly_batch_division_part_2(storage *poly,
                                           storage *result,
                                           const storage *powers,
                                           const uint currP,
                                           uint buckets_num,
                                           uint chunk_size,
                                           uint n) {
    uint gid = GET_GLOBAL_ID();

    if (gid >= buckets_num)
        return;
    if (gid == 0)
        return;
    if (gid * chunk_size >= n)
        return;
    
    storage p = powers[currP];
    
    storage acc = fd_q::get_zero();
    const storage pp = powers[currP * chunk_size];
    for (int z = 0; z < gid; ++z)
        acc = fd_q::add(fd_q::mul(acc, pp), result[z]);

    uint start = gid * chunk_size;
    uint end = min(start + chunk_size, n);

    for (uint j = start; j < end; ++j) {
        acc = fd_q::mul(acc, p);
        memory::store(poly + j, fd_q::add(acc, poly[j]));
    }
}

__global__ void poly_batch_division_part_3(storage *poly,
                                           storage *result,
                                           const storage *powers,
                                           const uint currP,
                                           uint buckets_num,
                                           uint chunk_size,
                                             uint n) {
    uint gid = GET_GLOBAL_ID();

    if (gid >= buckets_num)
        return;
    if (gid == 0)
        return;
    if (gid * chunk_size >= n)
        return;

    storage p = powers[currP];    
    storage acc = result[gid - 1];

    uint start = gid * chunk_size;
    uint end = min(start + chunk_size, n);

    for (uint j = start; j < end; ++j) {
        acc = fd_q::mul(acc, p);
        memory::store(poly + j, fd_q::add(acc, poly[j]));
    }
}

__global__ void poly_reverse(storage *poly, const storage *from, int n) 
{
    uint gid = GET_GLOBAL_ID();
    if (gid >= n)
        return;
    if (gid == 0)
        memory::store(poly + (n - 1 - gid), fd_q::get_zero());
    else
        memory::store(poly + (n - 1 - gid), memory::load(from + gid - 1));
}

static int sizes[100];
void compute_poly_division_kernel(storage *poly, storage *poly1, storage **result, storage* powers, const storage& p, 
                                  int buckets_num, int chunk_size, int deep, int chunk_size1, int N, cudaStream_t st)
{
    get_powers_kernel(powers, p, N, st);
    
    int pp = 1;
    int nextN = N;
    int bnum = 0;

    poly_batch_division_part_1<<<get_bl(buckets_num, 32), 32, 0, st>>>(poly, result[0], powers, pp, buckets_num, chunk_size, N);
    pp *= chunk_size;    
    nextN = nextN / chunk_size + ((nextN % chunk_size) != 0);

    for (int m = 0; m < deep; ++m)
    {
        sizes[m] = nextN;
        bnum = nextN / chunk_size1 + ((nextN % chunk_size1) != 0);
        if (m > 0)
            pp *= chunk_size1;
        poly_batch_division_part_1<<<get_bl(bnum, 32), 32, 0, st>>>(result[m], result[m + 1], powers, pp, bnum, chunk_size1, nextN);
        
        if (m != deep - 1)
            nextN = nextN / chunk_size1 + ((nextN % chunk_size1) != 0);
    }

    bnum = nextN / chunk_size1 + ((nextN % chunk_size1) != 0);
    poly_batch_division_part_2<<<get_bl(bnum, 32), 32, 0, st>>>(result[deep - 1], result[deep], powers, pp, bnum, chunk_size1, nextN);

    for (int m = (deep - 1) - 1; m >= 0; --m)
    {
        nextN = sizes[m];
        bnum = nextN / chunk_size1 + ((nextN % chunk_size1) != 0);

        pp /= chunk_size1;
        poly_batch_division_part_3<<<get_bl(bnum, 32), 32, 0, st>>>(result[m], result[m + 1], powers, pp, bnum, chunk_size1, nextN);
    }

    pp /= chunk_size;
    poly_batch_division_part_3<<<get_bl(buckets_num, 32), 32, 0, st>>>(poly, result[0], powers, pp, buckets_num, chunk_size, N);
    
    poly_reverse<<<get_bl(N, 64), 64, 0, st>>>(poly1, poly, N);
}

__device__ uint bitreverse(uint n, uint bits) {
    uint r = 0;
    for(int i = 0; i < bits; i++) {
        r = (r << 1) | (n & 1);
        n >>= 1;
    }
    return r;
}

// FFT algorithm is inspired from: http://www.bealto.com/gpu-fft_group-1.html
__global__ void poly_radix_fft(const storage *x, // Source buffer
                               storage* y, // Destination buffer
                               const storage* pq, // Precalculated twiddle factors
                               const storage* omegas, // [omega, omega^2, omega^4, ...]
                               uint n, // Number of elements
                               uint lgp, // Log2 of `p` (Read more in the link above)
                               uint deg, // 1=>radix2, 2=>radix4, 3=>radix8, ...
                               uint max_deg) // Maximum degree supported, according to `pq` and `omegas`
{

    uint __lid = threadIdx.x;
    uint lsize = 1 << deg >> 1; // get_local_size(0) == 1 << max_deg
    uint lid = __lid & (lsize - 1);
    uint loffset = (__lid - lid);
    uint index = (blockIdx.x << (max_deg - deg)) + (loffset >> (deg - 1));
    uint t = n >> deg;
    uint p = 1 << lgp;
    uint k = index & (p - 1);

    x += index;
    y += ((index - k) << deg) + k;

    uint count = 1 << deg; // 2^deg
    uint counth = count >> 1; // Half of count

    uint counts = count / lsize * lid;
    uint counte = counts + count / lsize;

    __shared__ storage uu[256];
    storage* u = uu;

    u += 2*loffset;
    __shared__ storage pq_shared[128];
    // load pq_shared
    pq_shared[threadIdx.x] = memory::load(pq + threadIdx.x);

    // Compute powers of twiddle
    uint t_power = (n >> lgp >> deg) * k;
    uint new_counts = counts + 1;
    storage tmp = fd_q::get_one();
    storage tmp_1 = fd_q::get_one();

    if (t_power != 0) {
        tmp = memory::load(omegas + t_power * counts);
        tmp_1 = memory::load(omegas + t_power * new_counts);
        u[counts] = fd_q::mul(tmp, memory::load(x + counts * t));
        u[counts + 1] = fd_q::mul(tmp_1, memory::load(x + new_counts * t));
    } else {
        u[counts] = memory::load(x + counts * t);
        u[counts + 1] = memory::load(x + new_counts * t);
    }
    __syncthreads();

    uint pqshift = max_deg - deg;
    for(uint rnd = 0; rnd < deg; rnd++) {
        uint bit = counth >> rnd;
        for(uint i = counts >> 1; i < counte >> 1; i++) {
          uint di = i & (bit - 1);
          uint i0 = (i << 1) - di;
          uint i1 = i0 + bit;
          tmp = u[i0];
          u[i0] = fd_q::add(u[i0], u[i1]);
          u[i1] = fd_q::sub(tmp, u[i1]);

          if(di != 0) 
              u[i1] = fd_q::mul(pq_shared[di << rnd << pqshift], u[i1]);
        }

        __syncthreads();
    }

    for(uint i = counts >> 1; i < counte >> 1; i++) {
        memory::store(y + i*p, u[bitreverse(i, deg)]);
        memory::store(y + (i+counth)*p, u[bitreverse(i + counth, deg)]);
    }
}

void compute_fft_kernel(storage* gpu_poly, storage* tmp_buffer, const storage* pq_buffer, const storage *omegas_buffer, const int n,
                        const int lgp, const int deg, const int max_deg, const int gws, const int lws, cudaStream_t st)
{
    poly_radix_fft<<<gws, lws, 0, st>>> (gpu_poly, tmp_buffer, pq_buffer, omegas_buffer, n, lgp, deg, max_deg);
}