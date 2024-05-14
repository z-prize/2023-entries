/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#include <gmp.h>
#include "mod_mult.h"

std::string mpz_to_string(const mpz_t &number) {
    int base = 10;
    char *buffer = new char[mpz_sizeinbase(number, base) + 2];
    mpz_get_str(buffer, base, number);
    std::string result(buffer);
    delete[] buffer;
    return result;
}

template<bls12_t CURVE>
ap_uint<384> to_mont(const ap_uint<384> &x) {
    return (x * base_field_t<CURVE>::R_MOD_P) % base_field_t<CURVE>::P;
}

template<bls12_t CURVE>
ap_uint<384> from_mont(const ap_uint<384> &x) {
    return (x * base_field_t<CURVE>::R_INV_MOD_P) % base_field_t<CURVE>::P;
}

int main() {
    mpz_t P_381, P_377;
    mpz_init_set_str(P_381,
                     "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab",
                     16);
    mpz_init_set_str(P_377,
                 "01ae3a4617c510eac63b05c06ca1493b1a22d9f300f5138f1ef3622fba094800170b5d44300000008508c00000000001",
                 16);

    gmp_randstate_t state;
    gmp_randinit_default(state);
    std::cout << "-- [common][la_modmul<bls12_381> random test] ----------------------------------" << std::endl;
    for (int i = 0; i < 50000; i++) {
        mpz_t tmp_a, tmp_b;
        mpz_init(tmp_a);
        mpz_init(tmp_b);
        mpz_urandomm(tmp_a, state, P_381);
        mpz_urandomm(tmp_b, state, P_381);
        ap_uint<384> a = ap_uint<384>(mpz_to_string(tmp_a).c_str());
        ap_uint<384> b = ap_uint<384>(mpz_to_string(tmp_b).c_str());
        ap_uint<384> c = (a * b) % base_field_t<bls12_381>::P;
        mpz_clear(tmp_a);
        mpz_clear(tmp_b);
        ap_uint<384> a_mont = to_mont<bls12_381>(a);
        ap_uint<384> b_mont = to_mont<bls12_381>(b);
        ap_uint<384> c_mont = mod_mult<bls12_381>::la_mod_mult(a_mont, b_mont);
        assert(from_mont(c_mont) == c);
    }

    std::cout << "PASSED" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;

    std::cout << "-- [common][la_modmul<bls12_377> random test] ----------------------------------" << std::endl;
    for (int i = 0; i < 50000; i++) {
        mpz_t tmp_a, tmp_b;
        mpz_init(tmp_a);
        mpz_init(tmp_b);
        mpz_urandomm(tmp_a, state, P_377);
        mpz_urandomm(tmp_b, state, P_377);
        ap_uint<384> a = ap_uint<384>(mpz_to_string(tmp_a).c_str());
        ap_uint<384> b = ap_uint<384>(mpz_to_string(tmp_b).c_str());
        ap_uint<384> c = (a * b) % base_field_t<bls12_377>::P;
        mpz_clear(tmp_a);
        mpz_clear(tmp_b);
        ap_uint<384> a_mont = to_mont<bls12_377>(a);
        ap_uint<384> b_mont = to_mont<bls12_377>(b);
        ap_uint<384> c_mont = mod_mult<bls12_377>::la_mod_mult(a_mont, b_mont);
        assert(from_mont(c_mont) == c);
    }

    std::cout << "PASSED" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;

    std::cout << "-- [common][ll_modmul<bls12_381> random test] ----------------------------------" << std::endl;

    for (int i = 0; i < 20000; i++) {
        mpz_t tmp_a, tmp_b;
        mpz_init(tmp_a);
        mpz_init(tmp_b);
        mpz_urandomm(tmp_a, state, P_381);
        mpz_urandomm(tmp_b, state, P_381);
        ap_uint<384> a = ap_uint<384>(mpz_to_string(tmp_a).c_str());
        ap_uint<384> b = ap_uint<384>(mpz_to_string(tmp_b).c_str());
        ap_uint<384> c = (a * b) % base_field_t<bls12_381>::P;
        mpz_clear(tmp_a);
        mpz_clear(tmp_b);
        ap_uint<384> a_mont = to_mont<bls12_381>(a);
        ap_uint<384> b_mont = to_mont<bls12_381>(b);
        ap_uint<384> c_mont = mod_mult<bls12_381>::ll_mod_mult(a_mont, b_mont);
        assert(from_mont(c_mont) == c);
    }

    std::cout << "PASSED" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;

    std::cout << "-- [common][ll_modmul<bls12_377> random test] ----------------------------------" << std::endl;

    for (int i = 0; i < 20000; i++) {
        mpz_t tmp_a, tmp_b;
        mpz_init(tmp_a);
        mpz_init(tmp_b);
        mpz_urandomm(tmp_a, state, P_377);
        mpz_urandomm(tmp_b, state, P_377);
        ap_uint<384> a = ap_uint<384>(mpz_to_string(tmp_a).c_str());
        ap_uint<384> b = ap_uint<384>(mpz_to_string(tmp_b).c_str());
        ap_uint<384> c = (a * b) % base_field_t<bls12_377>::P;
        mpz_clear(tmp_a);
        mpz_clear(tmp_b);
        ap_uint<384> a_mont = to_mont<bls12_377>(a);
        ap_uint<384> b_mont = to_mont<bls12_377>(b);
        ap_uint<384> c_mont = mod_mult<bls12_377>::ll_mod_mult(a_mont, b_mont);
        assert(from_mont(c_mont) == c);
    }

    std::cout << "PASSED" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
}