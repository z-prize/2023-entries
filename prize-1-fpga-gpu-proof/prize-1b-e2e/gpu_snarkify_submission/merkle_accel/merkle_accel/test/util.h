/***

Copyright (c) 2024, Ingonyama, Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#pragma once
#include <sstream>
#include "host.h"
#include "types.h"

inline void dump(st_t v) {
    printf("%016lx%016lx%016lx%016lx\n", v.l[3], v.l[2], v.l[1], v.l[0]);
}

inline void dump(st_mont_t v) {
    printf("m %016lx%016lx%016lx%016lx\n", v.l[3], v.l[2], v.l[1], v.l[0]);
}

inline void dump(storage<ff::bls12381_fp, false> v) {
    printf("%016lx%016lx%016lx%016lx%016lx%016lx\n", v.l[5], v.l[4], v.l[3], v.l[2], v.l[1], v.l[0]);
}

inline std::string to_string(st_t v) {
    char buffer[256];
    snprintf(
        buffer, sizeof(buffer),
        "%016lx%016lx%016lx%016lx", v.l[3], v.l[2], v.l[1], v.l[0]
    );
    return {buffer};
}

inline std::string to_string(storage<ff::bls12381_fp, false> v) {
    char buffer[256];
    snprintf(
        buffer, sizeof(buffer),
        "%016lx%016lx%016lx%016lx%016lx%016lx", v.l[5], v.l[4], v.l[3], v.l[2], v.l[1], v.l[0]
    );
    return {buffer};
}

inline st_t *alloc(const uint32_t size) {
    return static_cast<st_t *>(malloc(size * sizeof(st_t)));
}

inline affine_t *alloc_points(const uint32_t count) {
    return static_cast<affine_t *>(malloc(sizeof(affine_t) * count));
}

inline void store(const char *path, st_t *data, uint32_t count) {
    FILE *file = fopen(path, "w");

    if (file == NULL) {
        fprintf(stderr, "Failed to open '%s' for writing, aborting\n", path);
        exit(1);
    }
    fprintf(stderr, "Writing %s\n", path);
    for (uint32_t i = 0; i < count; i++) {
        fprintf(file, "%016lX%016lX%016lX%016lX\n", data[i].l[3], data[i].l[2], data[i].l[1], data[i].l[0]);
    }
    fclose(file);
}

inline affine_field_t result_for_point(const affine_t &pt) {
    affine_field_t res = pt.x;
    uint64_t doubled[6];

    if (pt.x.is_zero() && pt.y.is_zero()) {
        res.l[5] |= 1ull << 62;
    }

    host::mp_packed<ff::bls12381_fp>::add(doubled, pt.y.l, pt.y.l);
    if (host::mp_packed<ff::bls12381_fp>::compare(doubled, ff::bls12381_fp::const_storage_n.l) == 1) {
        res.l[5] |= 1ull << 63;
    }
    return res;
}

inline void copy(st_t *to, const st_t *from, const uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        to[i] = from[i];
    }
}


inline void commitment_challenges(
    impl_mont_t *challenges,
    const impl_mont_t challenge,
    const uint32_t commitment_count
) {
    impl_mont_t current = impl_mont_t::one();

    for (uint32_t i = 0; i < commitment_count; i++) {
        challenges[i] = current;
        current = current * challenge;
    }
}

inline void commitment_initialize(st_t *commitment, const uint32_t coefficient_count) {
    for (uint32_t i = 0; i < coefficient_count; i++) {
        commitment[i] = st_t::zero();
    }
}

inline void commitment_include(
    st_t *commitment,
    impl_mont_t challenge,
    st_t *polynomial,
    const uint32_t coefficient_count
) {
    for (int i = 0; i < coefficient_count; i++) {
        commitment[i] = commitment[i].mod_add(polynomial[i].mod_mul(challenge));
    }
}
