/***

Copyright (c) 2024, Snarkify, Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

#pragma once
#include <cstdlib>
#include "core.h"
#include "host.h"

typedef storage<ec::xy<ec::bls12381>, false> affine_t;
typedef storage<ff::bls12381_fp, false> affine_field_t;
typedef storage<ff::bls12381_fr, false> st_t;
typedef storage<ff::bls12381_fr, true> st_mont_t;
