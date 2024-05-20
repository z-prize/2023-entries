#ifndef __CURVE_EDWARDS_H__
#define __CURVE_EDWARDS_H__

#include <stdio.h>
#include "bls12_377_config.h"

namespace CURVE_EXED
{
    void precompute_exed_bases_points_for_64bit(
        CURVE_BLS12_377::affine_exed_64bit *bases_result,
        CURVE_BLS12_377::affine_exed_64bit *bases_origin,
        int count);
}

#endif