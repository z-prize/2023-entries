/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

#define ROOTS_6_3x3                     32
#define ROOTS_7                         (32 + 64)
#define ROOTS_8_2x6                     (32 + 64 + 128)
#define ROOTS_9_3x6                     (32 + 64 + 128 + 256)
#define ROOTS_14_7x7                    (32 + 64 + 128 + 256 + 512)
#define ROOTS_16_7x9                    (32 + 64 + 128 + 256 + 512 + 16384)
#define ROOTS_12                        (32 + 64 + 128 + 256 + 512 + 16384 + 65536)
#define ROOTS_22_LOW_10                 (32 + 64 + 128 + 256 + 512 + 16384 + 65536 + 4096)
#define ROOTS_25_LOW_13                 (32 + 64 + 128 + 256 + 512 + 16384 + 65536 + 4096 + 1024)
#define COSET_TO_MONTGOMERY_25_HIGH_12  (32 + 64 + 128 + 256 + 512 + 16384 + 65536 + 4096 + 1024 + 8192)
#define COSET_25_HIGH_12                (32 + 64 + 128 + 256 + 512 + 16384 + 65536 + 4096 + 1024 + 8192 + 4096)
#define COSET_25_LOW_13                 (32 + 64 + 128 + 256 + 512 + 16384 + 65536 + 4096 + 1024 + 8192 + 4096 + 4096)
#define ROOT_COUNT                      (32 + 64 + 128 + 256 + 512 + 16384 + 65536 + 4096 + 1024 + 8192 + 4096 + 4096 + 8192)

#include "InnerNTT.cu"
#include "NTT.cu"