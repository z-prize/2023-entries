/***

Copyright (c) 2024, Snarkify, Inc. and Ingonyama, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart
            Tony Wu

***/

#pragma once
typedef impl<ff::bls12381_fr, true> impl_mont_t;
typedef impl<ff::bls12381_fr, false> impl_t;

typedef host::ntt_context<ff::bls12381_fr> ntt_context_t;
typedef host::msm_context<ec::bls12381> msm_context_t;
typedef host::poly_context<ff::bls12381_fr> poly_context_t;

typedef struct {
    impl_mont_t alpha;
    impl_mont_t beta;
    impl_mont_t gamma;
    impl_mont_t delta;
    impl_mont_t epsilon;
    impl_mont_t z;
    impl_mont_t aw;
    impl_mont_t saw;
} challenges_t;

typedef struct {
    st_t left;
    st_t right;
    st_t hash;
} merkle_node_t;

typedef impl_mont_t poseidon_state_t[3];

typedef struct {
    impl_mont_t mds[3 * 3];
    impl_mont_t rc[63 * 3];
} poseidon_params_t;
