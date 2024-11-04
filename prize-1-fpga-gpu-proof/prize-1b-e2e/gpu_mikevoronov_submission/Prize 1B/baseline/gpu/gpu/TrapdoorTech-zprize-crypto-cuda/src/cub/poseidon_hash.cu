#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <map>
#include <vector>
#include <cassert>

#include "ff_config.cuh"
#include "ff_storage.cuh"
#include "ff_dispatch_st.cuh"
#include "memory.cuh"

#define SAFE_CALL(call) do { \
        cudaError err = call; \
        if (cudaSuccess != err) { \
                const char *errStr = cudaGetErrorString(err);\
                fprintf(stderr, "Cuda error %d in file '%s' in line %i : %s.\n", (int)err, __FILE__, __LINE__, errStr); \
                exit(-1); \
                } \
} while (0)

static int get_bl(const int count, const int th)
{
    return (count / th) + ((count % th) != 0);
}

void get_pi(void* res_, void* data_) {
    fd_q::storage& res = ((fd_q::storage*)res_)[0];
    fd_q::storage& data = ((fd_q::storage*)data_)[0];
    
    res = fd_q::neg(data);
}

HOST_DEVICE_INLINE size_t left_child_index(const size_t index) {
    return 2 * index + 1;
}

HOST_DEVICE_INLINE size_t right_child_index(const size_t index) {
    return 2 * index + 2;
}

HOST_DEVICE_INLINE
void print(const char* s, const fd_q::storage& elem) {
    unsigned long long* val = (unsigned long long*)elem.limbs;
    printf("%s %llu %llu %llu %llu\n", s, val[0], val[1], val[2], val[3]);
}

struct ArithmeticGate {
    size_t witness[3];
    int witness_size;
    
    //std::vector<std::pair<fd_q::storage, size_t>> fan_in_3;
    fd_q::storage mul_selector;
    fd_q::storage add_selectors[2];
    //fd_q::storage out_selector;
    fd_q::storage const_selector;

    fd_q::storage pi;
    int has_pi;

    HOST_DEVICE_INLINE ArithmeticGate() {
        mul_selector = fd_q::get_zero();
        add_selectors[0] = fd_q::get_zero();
        add_selectors[1] = fd_q::get_zero();
        //out_selector = fd_q::get_m_one();
        const_selector = fd_q::get_zero();
    }

    HOST_DEVICE_INLINE void fill_witness(size_t w_l, size_t w_r, size_t* w_o = NULL) {
        witness[0] = w_l;
        witness[1] = w_r;
        if (w_o) {
            witness[2] = *w_o;
            witness_size = 3;
        }
        else
            witness_size = 2;
    }


    HOST_DEVICE_INLINE void add(const fd_q::storage& q_l, const fd_q::storage& q_r) {
        add_selectors[0] = q_l;
        add_selectors[1] = q_r;
    }

    HOST_DEVICE_INLINE void constant(const fd_q::storage& q_c) {
        const_selector = q_c;     
    }

    HOST_DEVICE_INLINE void add_pi(const fd_q::storage* pi) {
        if (pi) {
            this->pi = *pi;
            has_pi = 1;
        }
        else
            has_pi = 0;
    }
};

extern fd_q::storage* get_pool(const int N, bool with_zero = false);

struct StandardComposer {
    /// Number of arithmetic gates in the circuit
    //size_t total = 0;
    size_t last_key;
    size_t last_gate;
    bool isGpu = false;
    
    fd_q::storage* public_inputs;
    
    // Witness vectors
    size_t* w_l = NULL;
    size_t* w_r = NULL;
    size_t* w_o = NULL;
    size_t* w_4 = NULL;

    size_t zero_var = 0;
    fd_q::storage* variables;
    //Permutation perm;
    
    HOST_DEVICE_INLINE
    size_t add_input(const fd_q::storage& s, size_t idx) {        
        variables[idx] = s;
        
        //unsigned long long *v = (unsigned long long *)s.limbs;
        //printf("%llu %llu %llu %llu\n", v[0],v[1],v[2],v[3]);
        return idx;
    }
    
    HOST_DEVICE_INLINE
    StandardComposer( ) { last_key = last_gate = 0; }
    
    HOST_DEVICE_INLINE
    StandardComposer(size_t* wl, size_t* wr, size_t* wo, size_t* w4, 
                     fd_q::storage* vars, fd_q::storage* pi, 
                     size_t last_key, size_t last_gate) : last_key(last_key), last_gate(last_gate) {
        w_l = wl;
        w_r = wr;
        w_o = wo;
        w_4 = w4;
        variables = vars;
        public_inputs = pi;
    }
    
    StandardComposer(int N = 0) {
        with_expected_size(N);
    }
    
    StandardComposer(int N, size_t* wl, size_t* wr, size_t* wo, size_t* w4, size_t len,
                     const fd_q::storage* vars, size_t varsLen) {
        with_expected_size(N);
                
        for (size_t z = 0; z < len; ++z) {
            w_l[z] = wl[z];
            w_r[z] = wr[z];
            w_o[z] = wo[z];
            w_4[z] = w4[z];
        }
        last_gate = len;

        for (size_t z = 0; z < varsLen; ++z)
            variables[z] = vars[z];
        
        last_key = varsLen;
    }
    
    //for GPU
    StandardComposer(const char *str) {
        isGpu = true;
    }
    
    //TODO: check N
    void copyFromHost(int N, const StandardComposer& host) {
        if (w_l == NULL)
        {
            SAFE_CALL(cudaMalloc((void**)&w_l, sizeof(size_t) * N));
            SAFE_CALL(cudaMalloc((void**)&w_r, sizeof(size_t) * N));
            SAFE_CALL(cudaMalloc((void**)&w_o, sizeof(size_t) * N));
            SAFE_CALL(cudaMalloc((void**)&w_4, sizeof(size_t) * N));
            
            SAFE_CALL(cudaMalloc((void**)&public_inputs, sizeof(fd_q::storage) * N));
            SAFE_CALL(cudaMalloc((void**)&variables, sizeof(fd_q::storage) * 2 * N));
        }
        
        SAFE_CALL(cudaMemset(public_inputs, 0, sizeof(fd_q::storage) * N));
        
        last_key = host.last_key;
        last_gate = host.last_gate;
        
        SAFE_CALL(cudaMemcpy(w_l, host.w_l, sizeof(size_t) * host.last_gate, cudaMemcpyHostToDevice));
        SAFE_CALL(cudaMemcpy(w_r, host.w_r, sizeof(size_t) * host.last_gate, cudaMemcpyHostToDevice));
        SAFE_CALL(cudaMemcpy(w_o, host.w_o, sizeof(size_t) * host.last_gate, cudaMemcpyHostToDevice));
        SAFE_CALL(cudaMemcpy(w_4, host.w_4, sizeof(size_t) * host.last_gate, cudaMemcpyHostToDevice));
        
        SAFE_CALL(cudaMemcpy(variables, host.variables, sizeof(fd_q::storage) * host.last_key, cudaMemcpyHostToDevice));
    }
    
    void clear() {
        if (!isGpu)
            return;
        //printf("clear comp\n");
        SAFE_CALL(cudaFree(w_l));
        SAFE_CALL(cudaFree(w_r));
        SAFE_CALL(cudaFree(w_o));
        SAFE_CALL(cudaFree(w_4));
        SAFE_CALL(cudaFree(variables));
        SAFE_CALL(cudaFree(public_inputs));
    }
    
    void with_expected_size(const size_t expected_size) {
        last_key = 0;
        last_gate = 0;

        w_l = new size_t[expected_size];
        w_r = new size_t[expected_size];
        w_o = new size_t[expected_size];
        w_4 = new size_t[expected_size];
        public_inputs = new fd_q::storage[expected_size];
        variables = new fd_q::storage[2 * expected_size];

        zero_var = add_witness_to_circuit_description(fd_q::get_zero());
    }

    size_t add_witness_to_circuit_description(const fd_q::storage& value) {
        auto var = add_input(value, last_key++);
        
        constrain_to_constant(var, NULL);
        return var;
    }

    void constrain_to_constant(size_t a, fd_q::storage* pi = NULL) {
        poly_gate(a, a, a, last_gate, pi);
    }

    HOST_DEVICE_INLINE
    void add_pi(size_t pos, const fd_q::storage& item) {
        public_inputs[pos] = item;
    }

    HOST_DEVICE_INLINE
    void poly_gate(const size_t a, const size_t b, const size_t c, size_t& gate_pos, fd_q::storage* pi = NULL) {

        w_l[gate_pos] = a;
        w_r[gate_pos] = b;
        w_o[gate_pos] = c;
        w_4[gate_pos] = zero_var;

        if (pi)
            add_pi(gate_pos, *pi);

        gate_pos++;
    }

    HOST_DEVICE_INLINE
    size_t arithmetic_gate(const ArithmeticGate& gate, size_t& gate_pos, size_t& var_pos) {
        if (gate.witness_size == 0) {
            //printf("Missing left and right wire witnesses\n");
            //exit(-1);
            return 0;
        }

        fd_q::storage q4;
        size_t w4;
        /*if (gate.fan_in_3.size()) {
            q4 = gate.fan_in_3[0].first;
            w4 = gate.fan_in_3[0].second;
        }
        else*/
        {
            q4 = fd_q::get_zero();
            w4 = zero_var;
        }
        
        w_4[gate_pos] = w4;
        auto& gate_witness = gate.witness;
        w_l[gate_pos] = gate_witness[0];
        w_r[gate_pos] = gate_witness[1];

        if (gate.has_pi) 
            add_pi(gate_pos, gate.pi);

        size_t c = 0;
        if (gate.witness_size == 3)
            c = gate_witness[2];
        else {
            fd_q::storage tmp;
            tmp = fd_q::mul(fd_q::mul(gate.mul_selector, variables[gate_witness[0]]), variables[gate_witness[1]]);
            tmp = fd_q::add(tmp, fd_q::mul(gate.add_selectors[0], variables[gate_witness[0]]));
            tmp = fd_q::add(tmp, fd_q::mul(gate.add_selectors[1], variables[gate_witness[1]]));
            tmp = fd_q::add(tmp, gate.const_selector);
            tmp = fd_q::add(tmp, fd_q::mul(q4, variables[w4]));
            if (gate.has_pi) 
                tmp = fd_q::add(tmp, gate.pi);
            //tmp = fd_q::mul(tmp, fd_q::neg(gate.out_selector));
            
            c = add_input(tmp, var_pos++);
        }
        w_o[gate_pos] = c;
        gate_pos++;

        return c;
    }

    HOST_DEVICE_INLINE
    void assert_equal(size_t a, size_t b, size_t& gate_pos) {
        poly_gate(a, b, zero_var, gate_pos, NULL);
    }

    HOST_DEVICE_INLINE
    void full_affine_transform_gate(size_t* res, const size_t* vars, const fd_q::storage m[3][3],
                                    const fd_q::storage current_round_key[3], size_t& gate_pos, size_t& var_pos) {
        auto& var0 = vars[0];
        auto& var1 = vars[1];
        auto& var2 = vars[2];
        
        fd_q::storage p[3];
        for (int z = 0; z < 3; ++z) {
            if (vars[z] != 0)
                p[z] = fd_q::pow5(variables[vars[z]]);
        }
        
        for (int z = 0; z < 3; ++z) {
            const fd_q::storage& sel0 = m[z][0];
            const fd_q::storage& sel1 = m[z][1];
            const fd_q::storage& sel2 = m[z][2];
            const fd_q::storage& sel3 = current_round_key[z];
            
            fd_q::storage w4_val = sel3;
            
            if (var0 != 0)
                w4_val = fd_q::add(w4_val, fd_q::mul(sel0, p[0]));

            if (var1 != 0)
                w4_val = fd_q::add(w4_val, fd_q::mul(sel1, p[1]));
            
            if (var2 != 0)
                w4_val = fd_q::add(w4_val, fd_q::mul(sel2, p[2]));

            size_t w4_var = add_input(w4_val, var_pos++);

            // add wires
            w_l[gate_pos] = var0;
            w_r[gate_pos] = var1;
            w_o[gate_pos] = w4_var;
            w_4[gate_pos] = var2;
            gate_pos++;

            res[z] = w4_var;
        }
    }
    
    HOST_DEVICE_INLINE
    void partial_affine_transform_gate(size_t* res, const size_t* vars, const fd_q::storage m[3][3], 
                                       const fd_q::storage* round_key, size_t& gate_pos, size_t& var_pos) {
        auto& var0 = vars[0];
        auto& var1 = vars[1];
        auto& var2 = vars[2];

        for (int z = 0; z < 3; ++z) {
            const fd_q::storage& sel0 = m[z][0];
            const fd_q::storage& sel1 = m[z][1];
            const fd_q::storage& sel2 = m[z][2];
            
            fd_q::storage w4_val = round_key[z];
            if (var0 != 0)
                w4_val = fd_q::add(w4_val, fd_q::mul(sel0, fd_q::pow5(variables[var0])));
            if (var1 != 0)
                w4_val = fd_q::add(w4_val, fd_q::mul(sel1, variables[var1]));
            if (var2 != 0)
                w4_val = fd_q::add(w4_val, fd_q::mul(sel2, variables[var2]));

            size_t w4_var = add_input(w4_val, var_pos++);

            // add wires
            w_l[gate_pos] = var0;
            w_r[gate_pos] = var1;
            w_o[gate_pos] = w4_var;
            w_4[gate_pos] = var2;

            gate_pos++;
            res[z] = w4_var;
        }
    }
};

struct MdsMatrices {
    fd_q::storage m[3][3];
};

struct PoseidonConstants {
    MdsMatrices mds_matrices;
    fd_q::storage* round_constants;
    fd_q::storage domain_tag;
    int full_rounds;
    int half_full_rounds;
    int partial_rounds;
    int round_size;

    HOST_DEVICE_INLINE
    PoseidonConstants (size_t full_rounds, size_t half_full_rounds, size_t partial_rounds,
                       size_t round_size, fd_q::storage* round_const, const fd_q::storage& domTag,
                       const fd_q::storage* matrix) :
        full_rounds(full_rounds), half_full_rounds(half_full_rounds), partial_rounds(partial_rounds), round_size(round_size) {

        round_constants = round_const;
        domain_tag = domTag;
        
        auto from = (fd_q::storage(*)[3])matrix;
        
        for (int z1 = 0; z1 < 3; ++z1)
            for (int z2 = 0; z2 < 3; ++z2)
                mds_matrices.m[z1][z2] = from[z1][z2];

        full_rounds = 8;
        half_full_rounds = 4;
        partial_rounds = 55;
    }
    
    PoseidonConstants(int gpu, const PoseidonConstants& host) {
        SAFE_CALL(cudaMalloc((void**)&round_constants, sizeof(fd_q::storage) * host.round_size));
        SAFE_CALL(cudaMemcpy(round_constants, host.round_constants, sizeof(fd_q::storage) * host.round_size, cudaMemcpyHostToDevice));
        
        for (int z1 = 0; z1 < 3; ++z1)
            for (int z2 = 0; z2 < 3; ++z2)
                mds_matrices.m[z1][z2] = host.mds_matrices.m[z1][z2];
            
        full_rounds = host.full_rounds;
        half_full_rounds = host.half_full_rounds;
        partial_rounds = host.partial_rounds;
        round_size = host.round_size;
        domain_tag = host.domain_tag;
    }
};

struct PoseidonZZRef {
    size_t constants_offset;
    size_t elements[3];
    size_t pos;
    const PoseidonConstants& constants;
    size_t last_k;
    size_t last_g;

    HOST_DEVICE_INLINE
    PoseidonZZRef(StandardComposer& composer, const PoseidonConstants& param, 
                  const size_t last_g, const size_t last_k_) : constants(param), last_g(last_g), last_k(last_k_) {
        constants_offset = 0;
        for (int z = 0; z < 3; ++z)
            elements[z] = composer.zero_var;
        elements[0] = composer.add_input(param.domain_tag, last_k++);
        pos = 1;
    }

    HOST_DEVICE_INLINE
    void input(const size_t input) {
        if (pos >= 3)
            return;

        elements[pos] = input;
        pos += 1;
    }

    HOST_DEVICE_INLINE
    size_t addi(StandardComposer& c, const size_t a, const fd_q::storage& b) {
        auto zero = c.zero_var;

        ArithmeticGate g;
        g.fill_witness(a, zero, NULL);
        g.add(fd_q::get_one(), fd_q::get_zero());
        g.constant(b);

        return c.arithmetic_gate(g, last_g, last_k);
    }
    
    HOST_DEVICE_INLINE
    void full_round(StandardComposer& c, size_t* state) {
        size_t shift = constants_offset;
        auto& pre_round_keys = constants.round_constants;

        size_t res[3];
        for (int z = 0; z < 3; ++z)
            res[z] = state[z];

        if (constants_offset == 0) {
            // first round
            res[0] = addi(c, res[0], pre_round_keys[shift + 0]);
            res[1] = addi(c, res[1], pre_round_keys[shift + 1]);
            res[2] = addi(c, res[2], pre_round_keys[shift + 2]);
        }
        
        fd_q::storage zero = fd_q::get_zero();
        fd_q::storage current_round_key[3];
        
        // Last round 
        if (shift + 3 >= constants.round_size) {
            for (int z = 0; z < 3; ++z)
                current_round_key[z] = zero;
        }
        else {
            for (int z = 0; z < 3; ++z)
                current_round_key[z] = pre_round_keys[shift + 3 + z];
        }

        auto& matrix = constants.mds_matrices.m;

        c.full_affine_transform_gate(state, res, matrix, current_round_key, last_g, last_k);
        constants_offset += 3;
    }

    HOST_DEVICE_INLINE
    void partial_round(StandardComposer& c, size_t* state) {
        size_t shift = constants_offset;
        auto& pre_round_keys = constants.round_constants;
            
        size_t res[3];
        for (int z = 0; z < 3; ++z)
            res[z] = state[z];

        auto& matrix = constants.mds_matrices.m;
                        
        c.partial_affine_transform_gate(state, res, matrix, pre_round_keys + shift + 3, last_g, last_k);
        constants_offset += 3;
    }

    HOST_DEVICE_INLINE
    size_t output_hash(StandardComposer& c) {
        for (int z = 0; z < constants.half_full_rounds; ++z)
            full_round(c, elements);

        for (int z = 0; z < constants.partial_rounds; ++z)
            partial_round(c, elements);

        for (int z = 0; z < constants.half_full_rounds; ++z)
            full_round(c, elements);

        return elements[1];
    }
};

HOST_DEVICE_INLINE
void assert_hash_constraints(StandardComposer& composer, const PoseidonConstants& param,
                             const size_t left, const size_t right, const size_t output,
                             size_t last_g, size_t last_k) {
    auto poseidon = PoseidonZZRef(composer, param, last_g, last_k);

    poseidon.input(left);
    poseidon.input(right);

    auto output_rec = poseidon.output_hash(composer);
    composer.assert_equal(output, output_rec, poseidon.last_g);
}

__global__ void gen_constraints_kernel(StandardComposer composer, PoseidonConstants hash_param,
                                       const size_t* leaf_node_vars, const size_t* non_leaf_node_vars, const fd_q::storage non_leaf_node_0,
                                       const int HEIGHT, const int N) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;
    
    //TODO: compiler bug for sm_80 and higher??
    if (gid == 0 && N == -1) {
        for (int z = 0; z < 1; ++z)
            print("", hash_param.mds_matrices.m[z][0]);
    }
    
    int last = HEIGHT - 2;
    size_t start_index = (1 << last) - 1;
    last--;
    
    auto upper_bound = left_child_index(start_index);
    const size_t full_assert_hash = 3 + (hash_param.full_rounds + hash_param.partial_rounds) * 3 + 1;

    size_t last_gate = composer.last_gate;
    size_t last_key = composer.last_key;
    
    size_t work = 0;
    size_t left, right, last_g, last_k, e;
    bool work_done = false;

    const size_t loop_size = upper_bound - start_index;
    for (size_t z = start_index + gid; z < upper_bound; z += N) {
        auto current_index = z;
        size_t index = z - start_index;

        auto left_leaf_index = left_child_index(current_index) - upper_bound;
        auto right_leaf_index = right_child_index(current_index) - upper_bound;

        if ((work + index) == gid) {
            left = leaf_node_vars[left_leaf_index];
            right = leaf_node_vars[right_leaf_index];
            e = non_leaf_node_vars[z];
            last_g = composer.last_gate + index * full_assert_hash;
            last_k = composer.last_key + index * full_assert_hash;
            work_done = true;
            
            break;
        }
    }

    composer.last_gate += full_assert_hash * loop_size;
    composer.last_key += full_assert_hash * loop_size;
    work += loop_size;

    if (!work_done) {
        for (int z = last; z >= 0; --z) {
            size_t start_index = (1 << z) - 1;
            auto upper_bound = left_child_index(start_index);

            const size_t loop_size = upper_bound - start_index;
            int lid = gid - work;
            for (size_t k = start_index + lid; k < upper_bound; k += N) {
                size_t index = k - start_index;

                if ((work + index) == gid) {
                    left = non_leaf_node_vars[left_child_index(k)];
                    right = non_leaf_node_vars[right_child_index(k)];
                    e = non_leaf_node_vars[k];
                    last_g = composer.last_gate + index * full_assert_hash;
                    last_k = composer.last_key + index * full_assert_hash;
                    work_done = true;
                    
                    break;
                }                
            }
            composer.last_gate += full_assert_hash * loop_size;
            composer.last_key += full_assert_hash * loop_size;
            work += loop_size;

            if (work_done)
                break;
        }
    }
    
    //printf("%d %d %d %d %d %d %d\n", (int)gid, (int)left, (int)right, (int)e, (int)last_g, (int)last_k, (int)work_done);
    assert_hash_constraints(composer, hash_param, left, right, e, last_g, last_k);
    
    if (gid == 0) {
        size_t root_node = non_leaf_node_vars[0];
        auto zero = composer.zero_var;

        ArithmeticGate gate;
        gate.fill_witness(root_node, zero, &zero);
        gate.add(fd_q::get_one(), fd_q::get_zero());

        fd_q::storage pi = fd_q::neg(non_leaf_node_0);
        gate.add_pi(&pi);
        
        last_gate += N * full_assert_hash;
        last_key += N * full_assert_hash;
        composer.arithmetic_gate(gate, last_gate, last_key);  
    }
}

//XXX: need to init with zero all arrays
__global__ void copy_scalars(fd_q::storage* wl, fd_q::storage* wr, fd_q::storage* wo, fd_q::storage* w4,
                             StandardComposer composer, unsigned N) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;
     
    const auto z = gid;
    auto key_l = composer.w_l[z];
    auto key_r = composer.w_r[z];
    auto key_o = composer.w_o[z];
    auto key_4 = composer.w_4[z];
    
    if (key_l != 0)
        wl[z] = memory::load(composer.variables + key_l);
    if (key_r != 0)
        wr[z] = memory::load(composer.variables + key_r);
    if (key_o != 0)
        wo[z] = memory::load(composer.variables + key_o);
    if (key_4 != 0)
        w4[z] = memory::load(composer.variables + key_4);
}

struct MerkleTree {
    std::vector<fd_q::storage> non_leaf_nodes;
    std::vector<fd_q::storage> leaf_nodes;
    
    size_t* non_leaf_node_vars_g = NULL;
    size_t* leaf_node_vars_g = NULL;

    size_t HEIGHT;
    
    MerkleTree(int H) : HEIGHT(H) 
    {
        
    }

    void init(int H, const fd_q::storage* nl_nodes, const fd_q::storage* l_nodes) 
    {
        int tt = 1 << (H - 1);
        if (HEIGHT != H || leaf_nodes.size() == 0)
        {
            leaf_nodes.resize(tt);
            non_leaf_nodes.resize(tt - 1);
            
            clear();
            SAFE_CALL(cudaMalloc((void**)&non_leaf_node_vars_g, sizeof(size_t) * (tt - 1)));
            SAFE_CALL(cudaMalloc((void**)&leaf_node_vars_g, sizeof(size_t) * tt)); 
        }
        else if (non_leaf_node_vars_g == NULL)
        {
            SAFE_CALL(cudaMalloc((void**)&non_leaf_node_vars_g, sizeof(size_t) * (tt - 1)));
            SAFE_CALL(cudaMalloc((void**)&leaf_node_vars_g, sizeof(size_t) * tt)); 
        }
        
        for (int z = 0; z < tt; ++z)
            leaf_nodes[z] = l_nodes[z];
        
        for (int z = 0; z < tt - 1; ++z)
            non_leaf_nodes[z] = nl_nodes[z];
    }
    
    void clear()
    {
        SAFE_CALL(cudaFree(non_leaf_node_vars_g));
        SAFE_CALL(cudaFree(leaf_node_vars_g));
    }
    
    void sync_mt_gpu(StandardComposer& composer, cudaStream_t st = 0) {
        size_t last_k = composer.last_key;
        SAFE_CALL(cudaMemcpyAsync(composer.variables + composer.last_key, leaf_nodes.data(), sizeof(fd_q::storage) * leaf_nodes.size(), cudaMemcpyHostToDevice, st));
        composer.last_key += leaf_nodes.size();
        
        SAFE_CALL(cudaMemcpyAsync(composer.variables + composer.last_key, non_leaf_nodes.data(), sizeof(fd_q::storage) * non_leaf_nodes.size(), cudaMemcpyHostToDevice, st));
        composer.last_key += non_leaf_nodes.size();
        
        std::vector<size_t> leaf_node_vars, non_leaf_node_vars;
        
        for (int z = 0; z < leaf_nodes.size(); ++z)
            leaf_node_vars.push_back(last_k++);

        for (int z = 0; z < non_leaf_nodes.size(); ++z)
            non_leaf_node_vars.push_back(last_k++);
        
        SAFE_CALL(cudaMemcpyAsync(leaf_node_vars_g, leaf_node_vars.data(), sizeof(size_t) * leaf_node_vars.size(), cudaMemcpyHostToDevice, st));
        SAFE_CALL(cudaMemcpyAsync(non_leaf_node_vars_g, non_leaf_node_vars.data(), sizeof(size_t) * non_leaf_node_vars.size(), cudaMemcpyHostToDevice, st));
    }
    
    void gen_constraints(StandardComposer& composer, const PoseidonConstants& hash_param) {
        std::vector<size_t> leaf_node_vars, non_leaf_node_vars;
        
        for (int z = 0; z < leaf_nodes.size(); ++z)
            leaf_node_vars.push_back(composer.add_input(leaf_nodes[z], composer.last_key++));

        for (int z = 0; z < non_leaf_nodes.size(); ++z)
            non_leaf_node_vars.push_back(composer.add_input(non_leaf_nodes[z], composer.last_key++));

        size_t root_node = non_leaf_node_vars[0];
        
        int last = HEIGHT - 2;
        size_t start_index = (1 << last) - 1;
        last--;
        
        auto upper_bound = left_child_index(start_index);
        const size_t full_assert_hash = 3 + (hash_param.full_rounds + hash_param.partial_rounds) * 3 + 1;

        size_t total = 0;
        for (size_t z = start_index; z < upper_bound; ++z) {
            total++;
            auto current_index = z;
            size_t e = non_leaf_node_vars[z];

            //printf("start = %d\n", current_index);
            auto left_leaf_index = left_child_index(current_index) - upper_bound;
            auto right_leaf_index = right_child_index(current_index) - upper_bound;
            
            assert_hash_constraints(composer, hash_param, 
                                    leaf_node_vars[left_leaf_index], 
                                    leaf_node_vars[right_leaf_index], 
                                    e, composer.last_gate, composer.last_key);
            
            composer.last_gate += full_assert_hash;
            composer.last_key += full_assert_hash;
        }

        for (int z = last; z >= 0; --z) {
            size_t start_index = (1 << z) - 1;
            auto upper_bound = left_child_index(start_index);

            for (size_t k = start_index, index = 0; k < upper_bound; ++k, ++index) {
                total++;
                size_t node = non_leaf_node_vars[k];
                //printf(" index = %d\n", index);
                assert_hash_constraints(composer, hash_param, 
                                        non_leaf_node_vars[left_child_index(index + start_index)], 
                                        non_leaf_node_vars[right_child_index(index + start_index)], 
                                        node, composer.last_gate, composer.last_key);
                
                composer.last_gate += full_assert_hash;
                composer.last_key += full_assert_hash;
            }
        }
        
        //printf("  total = %d\n", (int)total);
        auto zero = composer.zero_var;

        ArithmeticGate gate;
        gate.fill_witness(root_node, zero, &zero);
        gate.add(fd_q::get_one(), fd_q::get_zero());

        fd_q::storage pi = fd_q::neg(non_leaf_nodes[0]);
        gate.add_pi(&pi);

        composer.arithmetic_gate(gate, composer.last_gate, composer.last_key);        
    }
    
    void gen_constraints_gpu(StandardComposer& composer_g, const PoseidonConstants& hash_param_g, cudaStream_t st = 0) {
        const int N = (1 << (HEIGHT - 1)) - 1;
        gen_constraints_kernel<<<get_bl(N, 32), 32, 0, st>>>(composer_g, hash_param_g, leaf_node_vars_g, non_leaf_node_vars_g, non_leaf_nodes[0], HEIGHT, N);
    }
    
    void copy_scalars(StandardComposer& composer_g, const PoseidonConstants& hash_param,
                      fd_q::storage* wl, fd_q::storage* wr, fd_q::storage* wo, fd_q::storage* w4,
                      cudaStream_t st = 0) {
        const size_t full_assert_hash = 3 + (hash_param.full_rounds + hash_param.partial_rounds) * 3 + 1;
        const int N = ((1 << (HEIGHT - 1)) - 1) * full_assert_hash + 1 + composer_g.last_gate;
        //printf(" N = %d\n", N);
        ::copy_scalars<<<get_bl(N, 128), 128, 0, st>>>(wl, wr, wo, w4, composer_g, N);
    }
};

MerkleTree *mt = NULL;

StandardComposer *composer = NULL;
StandardComposer *composer_gpu = NULL;

PoseidonConstants *hash_param = NULL;
PoseidonConstants *hash_param_gpu = NULL;

#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

extern "C" void sync_hash_consts(size_t full_rounds, size_t half_full_rounds, size_t partial_rounds,
                                 size_t round_size, const void* round_const, const void* domTag, const void* matrix) {
    hash_param = new PoseidonConstants (full_rounds, half_full_rounds, partial_rounds, round_size, 
                                        (fd_q::storage*) round_const, ((const fd_q::storage*)domTag)[0], (fd_q::storage*)matrix);
                                        
    hash_param_gpu = new PoseidonConstants(0, *hash_param);
}

extern "C" void sync_composer(int H, const void* wl, const void* wr, const void* wo, const void* w4, size_t len,
                              const void* vars, size_t varsLen) {
    composer = new StandardComposer(1 << (H + 7), (size_t*)wl, (size_t*)wr, (size_t*)wo, (size_t*)w4, len, (const fd_q::storage*) vars, varsLen);
    
    if (composer_gpu == NULL)
        composer_gpu = new StandardComposer("gpu");

    composer_gpu->copyFromHost(1 << (H + 7), *composer);
    mt->sync_mt_gpu(*composer_gpu);
}

extern "C" void sync_mt(int H, const void* nl_nodes, const void* l_nodes) {
    if (mt == NULL)
        mt = new MerkleTree(H);
    
    mt->init(H, (const fd_q::storage*)nl_nodes, (const fd_q::storage*)l_nodes);
}

extern "C" void build_constraints() {
    auto time = high_resolution_clock::now();
    mt->gen_constraints(*composer, *hash_param);
    float elapsed = duration_cast<milliseconds>(high_resolution_clock::now() - time).count();
    printf("  c++ build = %.2f msec\n", elapsed);
    
    time = high_resolution_clock::now();
    const auto size = composer->last_gate;
    vector<fd_q::storage> wl_(size), wr_(size), wo_(size), w4_(size);
    for (size_t z = 0; z < size; ++z) {
        if (composer->w_l[z] != 0)
            wl_[z] = composer->variables[composer->w_l[z]];
        if (composer->w_r[z] != 0)
            wr_[z] = composer->variables[composer->w_r[z]];
        if (composer->w_o[z] != 0)
            wo_[z] = composer->variables[composer->w_o[z]];
        if (composer->w_4[z] != 0)
            w4_[z] = composer->variables[composer->w_4[z]];
    }
    elapsed = duration_cast<milliseconds>(high_resolution_clock::now() - time).count();
    printf("  c++ copy = %.2f msec\n", elapsed);
    
    fd_q::storage *wl, *wr, *wo, *w4;
    SAFE_CALL(cudaMalloc((void**)&wl, sizeof(fd_q::storage) * (1<<(mt->HEIGHT + 7))));
    SAFE_CALL(cudaMalloc((void**)&wr, sizeof(fd_q::storage) * (1<<(mt->HEIGHT + 7))));
    SAFE_CALL(cudaMalloc((void**)&wo, sizeof(fd_q::storage) * (1<<(mt->HEIGHT + 7))));
    SAFE_CALL(cudaMalloc((void**)&w4, sizeof(fd_q::storage) * (1<<(mt->HEIGHT + 7))));
    
    //printf(" N size = %d\n", size);
    
    time = high_resolution_clock::now();
    mt->gen_constraints_gpu(*composer_gpu, *hash_param_gpu);
    
    mt->copy_scalars(*composer_gpu, *hash_param, wl, wr, wo, w4);    
    SAFE_CALL(cudaDeviceSynchronize());
    elapsed = duration_cast<milliseconds>(high_resolution_clock::now() - time).count();
    printf("  c++ gpu build = %.2f msec\n", elapsed);
    
}

void* build_constraints_on_gpu(void *wl, void *wr, void *wo, void *w4, cudaStream_t st) {
    mt->gen_constraints_gpu(*composer_gpu, *hash_param_gpu, st);
    mt->copy_scalars(*composer_gpu, *hash_param, (fd_q::storage*)wl, (fd_q::storage*)wr, (fd_q::storage*)wo, (fd_q::storage*)w4, st);

    return composer_gpu->public_inputs;
}

static bool operator !=(const fd_q::storage& right, const fd_q::storage& left) {
    return !(fd_q::eq(right, left));
}

template<typename T>
static int checkOnGpu(const T* cpu, const T* gpu, int len) {
    int notEq = 0;
    T* host = new T[len];
    SAFE_CALL(cudaMemcpy(host, gpu, sizeof(T) * len, cudaMemcpyDeviceToHost));

    for (int z = 0; z < len; ++z) {
        if (host[z] != cpu[z])
            notEq++;
    }
    delete []host;
    
    return notEq;
}

extern "C" void check_constraints(const void *vars, const int len,
                                  const void *w0, const void *w1,
                                  const void *w2, const void *w3, const int lenW) {
    const fd_q::storage* rust = (const fd_q::storage*)vars;
    printf("%d %d\n", (int)composer->last_key, len);
    int notEq = 0;
    for (int z = 0; z < len; ++z) {
        if (!(fd_q::eq(composer->variables[z], rust[z])))
            notEq++;
    }    
    printf("not eq %d\n", notEq);
    
    const fd_q::storage* rustWl = (const fd_q::storage*)w0;
    const fd_q::storage* rustWr = (const fd_q::storage*)w1;
    const fd_q::storage* rustWo = (const fd_q::storage*)w2;
    const fd_q::storage* rustW4 = (const fd_q::storage*)w3;
    
    printf("%d %d\n", (int)composer->last_gate, lenW);
    const auto size = composer->last_gate;
    vector<fd_q::storage> wl(size), wr(size), wo(size), w4(size);
    for (size_t z = 0; z < size; ++z) {
        if (composer->w_l[z] != 0)
            wl[z] = composer->variables[composer->w_l[z]];
        else
            wl[z] = fd_q::get_zero();
        
        if (composer->w_r[z] != 0)
            wr[z] = composer->variables[composer->w_r[z]];
        else
            wr[z] = fd_q::get_zero();
        
        if (composer->w_o[z] != 0)
            wo[z] = composer->variables[composer->w_o[z]];
        else
            wo[z] = fd_q::get_zero();
        
        if (composer->w_4[z] != 0)
            w4[z] = composer->variables[composer->w_4[z]];
        else
            w4[z] = fd_q::get_zero();
    }
    
    for (size_t z = 0; z < size; ++z) {
        if (!(fd_q::eq(wl[z], rustWl[z])))
            notEq++;
        if (!(fd_q::eq(wr[z], rustWr[z])))
            notEq++;
        if (!(fd_q::eq(wo[z], rustWo[z])))
            notEq++;
        if (!(fd_q::eq(w4[z], rustW4[z])))
            notEq++;
    }
    printf("not eq sc %d\n", notEq);
    
    notEq = checkOnGpu(composer->w_l, composer_gpu->w_l, composer->last_gate);
    printf("not eq on gpu wl %d\n", notEq);
    notEq = checkOnGpu(composer->w_r, composer_gpu->w_r, composer->last_gate);
    printf("not eq on gpu wr %d\n", notEq);
    notEq = checkOnGpu(composer->w_o, composer_gpu->w_o, composer->last_gate);
    printf("not eq on gpu wo %d\n", notEq);
    notEq = checkOnGpu(composer->w_4, composer_gpu->w_4, composer->last_gate);
    printf("not eq on gpu w4 %d\n", notEq);
    
    notEq = checkOnGpu(composer->variables, composer_gpu->variables, composer->last_key);
    printf("not eq on gpu vars %d\n", notEq);
}
