#pragma once

#include "ff_config.cuh"
#include "ff_storage.cuh"
#include "ff_dispatch_st.cuh"

static bool operator !=(const fd_q::storage& right, const fd_q::storage& left) {
    return !(fd_q::eq(right, left));
}

#define Fq_limb uint
#define Fq_LIMBS 12
#define Fq_LIMB_BITS 32
typedef struct { Fq_limb val[Fq_LIMBS]; } Fq;

typedef struct { Fq val[2]; } affine;
typedef struct { Fq val[2]; Fq_limb dummy[2]; } affineG1;
typedef struct { Fq val[4]; } point_xyzz;
typedef struct { Fq val[3]; } point_jacobian;
