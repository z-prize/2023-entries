/***

Copyright (c) 2024, Ingonyama, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#include <iostream>
#include <cstdlib>
#include <cstring>

#include "merkle_accel.h"

void *ffi_init(const uint64_t *ffi_commit_key, prover_key_t ffi_prover_key) {
    return nullptr;
}

void ffi_round_0(void *ctx, const uint64_t *ffi_blinding_factors, const uint64_t *ffi_non_leaf_nodes,
                 const uint64_t *ffi_leaf_nodes) {
}

round_1_res_t ffi_round_1(void *ctx) {
    return {
        {
            0xb7954866ad64c162, 0x29cd1d083f46a6e4, 0x1cb428418da1640a,
            0x0afad7a4b44c8559, 0x8e7cfcef925c6f7b, 0x8de523e2f09d5603
        },
        {
            0x28f8f4cab1eabfe0, 0x7e67ab262eb76716, 0x3cc1448924bbc9d0,
            0xd2860bc92d70aa72, 0x3e05a6369563c462, 0x98a61e4239edbbcd
        },
        {
            0x9f8c94e35ebb90d4, 0x734cc5836c80595d, 0xab560704406ad15b,
            0x7baf0c3a954b8796, 0x0bc1451d13db384d, 0x0d9a664f5afdc484
        },
        {
            0xaeb409bc3871c16f, 0x2c7e35ba837aa0e9, 0xdde379dd6b883cfe,
            0x6ac39808aefbab21, 0x944502ee6e78fb1b, 0x8b7693d8d19c33a0
        }
    };
}

round_3_res_t ffi_round_3(void *ctx, const uint64_t *ffi_beta, const uint64_t *ffi_gamma) {
    return {
        {
            0xbecbc5e4ac68efad, 0x27572ddbf916faf3, 0x1bb1f396bdf3c49c,
            0x2f39a50895af81a1, 0x32f40c746b773474, 0x131cf98c7f790db0
        }
    };
}

round_4a_res_t ffi_round_4a(void *ctx, const uint64_t *ffi_alpha, const uint64_t *ffi_beta, const uint64_t *ffi_gamma) {
    return {
        {
            0x835fcd82f72b46ae, 0xf23469d75d88c538, 0x7c40c4279a163579,
            0xcdea8e9298d60841, 0xbc6165af55dc3e93, 0x946d84c02a8403ae
        },
        {
            0x5acace0a0bc139f4, 0xac10636107198d74, 0x3222e4f5978f2e09,
            0xa82583eb695839d3, 0x7ff9773378864ea8, 0x041ac34a4d34638d
        },
        {
            0x282e1f3f6f749e10, 0x8621289eef95edcc, 0x2edad49d61b70745,
            0xeb1d96b5e2c26a12, 0xadfbefbf2b6c371a, 0x150b5a9ac12cb237
        },
        {
            0x785084b42eee3505, 0xf79cedaa870eae15, 0x63ddd6faec4d810f,
            0x0845f85d714f64cd, 0x152fea2491c58656, 0x06a2191d0069f6a5
        },
        {
            0x11baf48f2d82a870, 0xfd5c995cf5c4cf17, 0x5bb5126411e83fe1,
            0xb7cd7eeb581b2d7f, 0xb5e918b0ec5dd080, 0x0d18951734f5401f
        },
        {
            0x3bc1acd13442affe, 0x60e8df9a22150022, 0x8a41679a3230c7b7,
            0x33dc27e74443fc59, 0xb20e5be92e95862e, 0x0d0d761f6a99feaa
        },
        {
            0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
            0x0000000000000000, 0x0000000000000000, 0x4000000000000000
        },
        {
            0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
            0x0000000000000000, 0x0000000000000000, 0x4000000000000000
        }
    };
}

round_4b_res_t ffi_round_4b(void *ctx, const uint64_t *ffi_alpha, const uint64_t *ffi_beta, const uint64_t *ffi_gamma,
                            const uint64_t *ffi_z_challenge) {
    return {
        {0xb1a71724bcac64ba, 0x1c2a68bfe4b7a278, 0xacd858adfbd2b6fc, 0x1b72b3c1b1fe2382},
        {0x91c042ce7e48d2d5, 0x168f6dabf2fdfe89, 0x5925ed8ee6076dc6, 0x53ccf83e3a7b634a},
        {0xdf7566d6bc09c7c4, 0x5298119720dbde92, 0x94a759045c5cf32c, 0x0dba4a0e3ce26548},
        {0x7272b6c4c9fca53b, 0xd05057877dcd75cd, 0xad5927e63301cbe4, 0x0a71f8f91bd92844},
        {0x66d5e6e6112eaef2, 0xf5f677d56b3cad3c, 0xe155681b20793001, 0x18ba49111abcf66a},
        {0x1ed69e9ad2c66aa2, 0x4d6bec18490a7aa5, 0x74e7e5c6eba42bb5, 0x3df7f708cc2fa49a},
        {0x548d0fca194464fe, 0xd26e93446a550385, 0xad1c379c1af9c3bc, 0x0297a21aec934a16},
        {0xa57beca353c45434, 0xcce588749a83d39f, 0xd688db259ce16cea, 0x2edec6d9423e3efb},
        {0xd5b7d9ffe2b84167, 0x736e94e332d0f2a6, 0x9b7dab38734820e3, 0x1af549e206f96a63},
        {0x64c5a5a52234542f, 0x407365adf614b328, 0xc5895b40d54ec767, 0x40ac43da5bf32967},
        {0x700f7cbb049d32dd, 0x0f2421d7bd283990, 0xbd794412b7f87456, 0x636c4e1fb0e3840f},
        {0xcbea99590befa452, 0x7db7ad89c8faedc3, 0x428c3bd738f59f45, 0x454f0c7b8e3c877c},
        {0xb68c45d535e28230, 0x1fcb8568f5f5c90c, 0xa5737e801094587a, 0x6a802ad6eac7d3b5},
        {0xe3afc11e119d81cb, 0x4ce007d6d8b2fb43, 0x60170273cbec5ff8, 0x109a13797a0808b3},
        {0xc71ffbc2219fcb82, 0x5b0934c37bf54758, 0x5991af27c02c5cd8, 0x4a2dfab2ba044231},
        {0x2bbe062e04d40894, 0x110c7fb4acd6a11e, 0x90bf42bc9aa091e3, 0x2394101b6cb9c994},
        {0x0fef83dc2dd7fb0d, 0xc6a0e0f3600ec4ae, 0xf2100f35de507465, 0x03539a80c7f3d0d6},
        {0x7535f683aae52c39, 0xd3ef613c084f70b2, 0xece0332adb854735, 0x6519e9bfa021c20c}
    };
}

round_5_res_t ffi_round_5(void *ctx, const uint64_t *ffi_z_challenge, const uint64_t *ffi_aw_challenge,
                          const uint64_t *ffi_saw_challenge) {
    return {
        {
            0x4c8295c2fe477070, 0x2f128dce94483676, 0xb1b952824b21e41d,
            0x5c25d8650e652b36, 0x769f70e56c1527ae, 0x869e44a7a681585d
        },
        {
            0xe26c89e674a9e82f, 0xdf4c2b75e4400f7f, 0x95ace99fbb2e20d3,
            0x776f3d782b3dda5a, 0x4455885f07e2aebe, 0x98443bead7f528a0
        }
    };
}
