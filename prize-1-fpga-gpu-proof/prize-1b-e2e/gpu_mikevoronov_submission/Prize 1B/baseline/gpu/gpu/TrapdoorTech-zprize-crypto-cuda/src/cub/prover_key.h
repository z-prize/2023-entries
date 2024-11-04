#pragma once

#include "types.h"
#include "common.h"
#include "memory.h"

struct Arithmetic {
    storage* q_m[2] = { nullptr, nullptr };
    storage* q_l[2] = { nullptr, nullptr };
    storage* q_r[2] = { nullptr, nullptr };
    storage* q_o[2] = { nullptr, nullptr };
    storage* q_4[2] = { nullptr, nullptr };
    storage* q_hl[2] = { nullptr, nullptr };
    storage* q_hr[2] = { nullptr, nullptr };
    storage* q_h4[2] = { nullptr, nullptr };
    storage* q_c[2] = { nullptr, nullptr };
    storage* q_arith[2] = { nullptr, nullptr };

    void check() {
       for (int z = 0; z < 2; ++z) {
           if (q_m[z] != nullptr) printf("not %d %d\n", z, __LINE__);
           if (q_l[z] != nullptr) printf("not %d %d\n", z, __LINE__);
           if (q_r[z] != nullptr) printf("not %d %d\n", z, __LINE__);
           if (q_o[z] != nullptr) printf("not %d %d\n", z, __LINE__);
           if (q_4[z] != nullptr) printf("not %d %d\n", z, __LINE__);
           if (q_hl[z] != nullptr) printf("not %d %d\n", z, __LINE__);
           if (q_hr[z] != nullptr) printf("not %d %d\n", z, __LINE__);
           if (q_h4[z] != nullptr) printf("not %d %d\n", z, __LINE__);
           if (q_c[z] != nullptr) printf("not %d %d\n", z, __LINE__);
           if (q_arith[z] != nullptr) printf("not %d %d\n", z, __LINE__);
       }
    }
    
    void reg_host(const int N) {
        const int domN = 8 * N;
        
        reg(0, N);
        reg(1, domN);
    }
    
    void copy_from_host(const int N, const Arithmetic& host, 
                        cudaStream_t st = 0, cudaStream_t st1 = 0) {
        const int domN = 8 * N;
        
        const int sizes[2] = { N, domN };
        cudaStream_t streams[2] = { st, st1 };
        
        copy(host, sizes, streams);
    }
    
    void allocate_data(const int N) {
        const int domN = 8 * N;
        const int sizes[2] = { N, domN };

        for (int z = 0; z < 2; ++z) {
            //q_m[z] = get_pool(sizes[z]);
            
            /*q_l[z] = get_pool(sizes[z]);
            q_r[z] = get_pool(sizes[z]);
            q_o[z] = get_pool(sizes[z]);
            q_4[z] = get_pool(sizes[z]);
            q_hl[z] = get_pool(sizes[z]);
            q_hr[z] = get_pool(sizes[z]);
            q_h4[z] = get_pool(sizes[z]);
            q_c[z] = get_pool(sizes[z]);
            q_arith[z] = get_pool(sizes[z]);*/
            
            q_l[z] = get_device<storage>(sizes[z]);
            q_r[z] = get_device<storage>(sizes[z]);
            q_o[z] = get_device<storage>(sizes[z]);
            q_4[z] = get_device<storage>(sizes[z]);
            q_hl[z] = get_device<storage>(sizes[z]);
            q_hr[z] = get_device<storage>(sizes[z]);
            q_h4[z] = get_device<storage>(sizes[z]);
            q_c[z] = get_device<storage>(sizes[z]);
            q_arith[z] = get_device<storage>(sizes[z]);
        }
    }

    void release_data(const int N, const int elem) {
        return;
        const int domN = 8 * N;
        const int sizes[2] = { N, domN };
        
        release_pool(q_m[elem], sizes[elem]);
        release_pool(q_l[elem], sizes[elem]);
        release_pool(q_r[elem], sizes[elem]);
        release_pool(q_o[elem], sizes[elem]);
        release_pool(q_4[elem], sizes[elem]);
        release_pool(q_hl[elem], sizes[elem]);
        release_pool(q_hr[elem], sizes[elem]);
        release_pool(q_h4[elem], sizes[elem]);
        release_pool(q_c[elem], sizes[elem]);
        release_pool(q_arith[elem], sizes[elem]);
    }
private:
    void reg(const int elem, const int N) {
        register_host(q_m[elem], N);
        register_host(q_l[elem], N);
        register_host(q_r[elem], N);
        register_host(q_o[elem], N);
        register_host(q_4[elem], N);
        register_host(q_hl[elem], N);
        register_host(q_hr[elem], N);
        register_host(q_h4[elem], N);
        register_host(q_c[elem], N);
        register_host(q_arith[elem], N);
    }
    
    void copy(const Arithmetic& host, const int sizes[2], cudaStream_t st[2]) {
        for (int z = 1; z >= 0; --z) {
            copy_to_device(q_m[z], host.q_m[z], sizes[z], st[z]);
            copy_to_device(q_l[z], host.q_l[z], sizes[z], st[z]);
            copy_to_device(q_r[z], host.q_r[z], sizes[z], st[z]);
            copy_to_device(q_o[z], host.q_o[z], sizes[z], st[z]);
            copy_to_device(q_4[z], host.q_4[z], sizes[z], st[z]);
            copy_to_device(q_hl[z], host.q_hl[z], sizes[z], st[z]);
            copy_to_device(q_hr[z], host.q_hr[z], sizes[z], st[z]);
            copy_to_device(q_h4[z], host.q_h4[z], sizes[z], st[z]);
            copy_to_device(q_c[z], host.q_c[z], sizes[z], st[z]);
            copy_to_device(q_arith[z], host.q_arith[z], sizes[z], st[z]);
        }
    }
};

struct Permutation {
    storage* linear_eval = nullptr;
    storage* left_sigma[2] = { nullptr, nullptr };
    storage* right_sigma[2] = { nullptr, nullptr };
    storage* out_sigma[2] = { nullptr, nullptr };
    storage* fourth_sigma[2] = { nullptr, nullptr };
    
    void check() {
       if (linear_eval != nullptr) printf("not %d\n", __LINE__);
       for (int z = 0; z < 2; ++z) {
           if (left_sigma[z] != nullptr) printf("not %d %d\n", z, __LINE__);
           if (right_sigma[z] != nullptr) printf("not %d %d\n", z, __LINE__);
           if (out_sigma[z] != nullptr) printf("not %d %d\n", z, __LINE__);
           if (fourth_sigma[z] != nullptr) printf("not %d %d\n", z, __LINE__);
       }
    }
    
    void reg_host(const int N) {
        const int domN = 8 * N;
        
        reg(0, N);
        reg(1, domN);
        
        register_host(linear_eval, domN);
    }

    void copy_from_host(const int N, const Permutation& host, 
                        cudaStream_t st = 0, cudaStream_t st1 = 0) {
        const int domN = 8 * N;
        
        const int sizes[2] = { N, domN };
        cudaStream_t streams[2] = { st, st1 };
        
        copy_to_device(linear_eval, host.linear_eval, domN, st1);
        copy(host, sizes, streams);
    }
    
    void allocate_data(const int N) {
        const int domN = 8 * N;
        const int sizes[2] = { N, domN };
        
        //linear_eval = get_pool(domN);
        linear_eval = get_device<storage>(domN);
        for (int z = 0; z < 2; ++z)
            allocate_sigma(sizes[z], z);
    }

    void allocate_sigma(const int N, const int elem) {
        /*left_sigma[elem] = get_pool(N);
        right_sigma[elem] = get_pool(N);
        out_sigma[elem] = get_pool(N);
        fourth_sigma[elem] = get_pool(N);*/
        
        left_sigma[elem] = get_device<storage>(N);
        right_sigma[elem] = get_device<storage>(N);
        out_sigma[elem] = get_device<storage>(N);
        fourth_sigma[elem] = get_device<storage>(N);
    }
    
    void release_data(const int N, const int elem) {
        return;
        const int domN = 8 * N;
        const int sizes[2] = { N, domN };
        
        if (elem == 1) 
            release_pool(linear_eval, sizes[elem]);

        release_pool(left_sigma[elem], sizes[elem]);
        release_pool(right_sigma[elem], sizes[elem]);
        release_pool(out_sigma[elem], sizes[elem]);
        release_pool(fourth_sigma[elem], sizes[elem]);
    }
private:
    void reg(const int elem, const int N) {
        register_host(left_sigma[elem], N);
        register_host(right_sigma[elem], N);
        register_host(out_sigma[elem], N);
        register_host(fourth_sigma[elem], N);
    }
    
    void copy(const Permutation& host, const int sizes[2], cudaStream_t st[2]) {
        for (int z = 1; z >= 0; --z) {
            copy_to_device(left_sigma[z], host.left_sigma[z], sizes[z], st[z]);
            copy_to_device(right_sigma[z], host.right_sigma[z], sizes[z], st[z]);
            copy_to_device(out_sigma[z], host.out_sigma[z], sizes[z], st[z]);
            copy_to_device(fourth_sigma[z], host.fourth_sigma[z], sizes[z], st[z]);
        }
    }
};

struct Prover_key {
    int N = 0;

    storage* v_h = nullptr;
    Arithmetic arith;
    Permutation perm;
    
    void reg_host() {
        const int domN = 8 * N;
        register_host(v_h, domN);
        
        arith.reg_host(N);
        perm.reg_host(N);
    }
    
    void copy_from_host(const Prover_key& host,
                        cudaStream_t st = 0, cudaStream_t st1 = 0) {
        if (N == 0) 
            return;
        const int domN = 8 * N;
        
        copy_to_device(v_h, host.v_h, domN, st1);
        perm.copy_from_host(N, host.perm, st, st1);
        arith.copy_from_host(N, host.arith, st, st1);
    }
    
    void allocate_data() {
        if (N == 0) 
            return;
        const int domN = 8 * N;
        //v_h = get_pool(domN);
        v_h = get_device<storage>(domN);
        
        arith.allocate_data(N);
        perm.allocate_data(N);
    }
    
    void release_data() {
        return;
        const int domN = 8 * N;
        release_pool(v_h, domN);
    }
    
    void check () {
        if (v_h) printf("not %d\n", __LINE__);
        
        arith.check();
        perm.check();
    }
};

static Prover_key prover_host, prover_gpu;
static Permutation perm_gpu;

extern "C" int getExpectedDomainSize() {
    return prover_host.N;
}     
extern "C" void 
     registerAllArrays  (const int N,
                         const void* prov_v_h, 
                         
                         const void* prov_arith_q_m_1, 
                         const void* prov_arith_q_l_1, 
                         const void* prov_arith_q_r_1, 
                         const void* prov_arith_q_o_1, 
                         const void* prov_arith_q_4_1, 
                         const void* prov_arith_q_hl_1, 
                         const void* prov_arith_q_hr_1, 
                         const void* prov_arith_q_h4_1, 
                         const void* prov_arith_q_c_1, 
                         const void* prov_arith_q_arith_1,
                         
                         const void* prov_perm_linear, 
                         const void* prov_perm_left_sig_1, 
                         const void* prov_perm_right_sig_1, 
                         const void* prov_perm_out_sig_1, 
                         const void* prov_perm_fourth_sig_1,
                         
                         const void* prov_arith_q_arith_0, 
                         const void* prov_arith_q_c_0, 
                         const void* prov_arith_q_l_0, 
                         const void* prov_arith_q_r_0, 
                         const void* prov_arith_q_hl_0, 
                         const void* prov_arith_q_hr_0,
                         const void* prov_arith_q_h4_0, 
                         const void* prov_arith_q_m_0, 
                         const void* prov_arith_q_4_0, 
                         const void* prov_arith_q_o_0,
                         
                         const void* prov_perm_left_sig_0, 
                         const void* prov_perm_right_sig_0,
                         const void* prov_perm_out_sig_0, 
                         const void* prov_perm_fourth_sig_0)
{
    prover_host.N = N;
    prover_gpu.N = N;
    
    prover_host.v_h = (storage*)prov_v_h;

    prover_host.arith.q_l[0] = (storage*)prov_arith_q_l_0;
    prover_host.arith.q_l[1] = (storage*)prov_arith_q_l_1;
    
    //prover_host.arith.q_m[0] = (storage*)prov_arith_q_m_0;
    //prover_host.arith.q_m[1] = (storage*)prov_arith_q_m_1;
    
    prover_host.arith.q_r[0] = (storage*)prov_arith_q_r_0;
    prover_host.arith.q_r[1] = (storage*)prov_arith_q_r_1;
    
    prover_host.arith.q_o[0] = (storage*)prov_arith_q_o_0;
    prover_host.arith.q_o[1] = (storage*)prov_arith_q_o_1;
    
    prover_host.arith.q_4[0] = (storage*)prov_arith_q_4_0;
    prover_host.arith.q_4[1] = (storage*)prov_arith_q_4_1;
    
    prover_host.arith.q_hl[0] = (storage*)prov_arith_q_hl_0;
    prover_host.arith.q_hl[1] = (storage*)prov_arith_q_hl_1;
    
    prover_host.arith.q_hr[0] = (storage*)prov_arith_q_hr_0;
    prover_host.arith.q_hr[1] = (storage*)prov_arith_q_hr_1;
    
    prover_host.arith.q_h4[0] = (storage*)prov_arith_q_h4_0;
    prover_host.arith.q_h4[1] = (storage*)prov_arith_q_h4_1;
    
    prover_host.arith.q_c[0] = (storage*)prov_arith_q_c_0;
    prover_host.arith.q_c[1] = (storage*)prov_arith_q_c_1;
    
    prover_host.arith.q_arith[0] = (storage*)prov_arith_q_arith_0;
    prover_host.arith.q_arith[1] = (storage*)prov_arith_q_arith_1;
    
    prover_host.perm.linear_eval = (storage*)prov_perm_linear;
    
    prover_host.perm.left_sigma[0] = (storage*)prov_perm_left_sig_0;
    prover_host.perm.left_sigma[1] = (storage*)prov_perm_left_sig_1;
    
    prover_host.perm.right_sigma[0] = (storage*)prov_perm_right_sig_0;
    prover_host.perm.right_sigma[1] = (storage*)prov_perm_right_sig_1;
    
    prover_host.perm.out_sigma[0] = (storage*)prov_perm_out_sig_0;
    prover_host.perm.out_sigma[1] = (storage*)prov_perm_out_sig_1;
    
    prover_host.perm.fourth_sigma[0] = (storage*)prov_perm_fourth_sig_0;
    prover_host.perm.fourth_sigma[1] = (storage*)prov_perm_fourth_sig_1;
    
    prover_host.reg_host();
}
   
extern "C" void copyArrays ()
{
    prover_gpu.allocate_data();
    prover_gpu.copy_from_host(prover_host);
    
    SAFE_CALL(cudaDeviceSynchronize());
}