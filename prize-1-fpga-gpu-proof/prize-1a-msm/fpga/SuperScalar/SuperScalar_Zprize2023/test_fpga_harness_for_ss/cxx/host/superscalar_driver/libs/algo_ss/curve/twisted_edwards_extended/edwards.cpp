#include <stdio.h>
#include <omp.h>
#include "config.h"
#include "bls12_377_config.h"

namespace CURVE_EXED
{

// 64bit
    void precompute_exed_point_for_64bit(
        CURVE_BLS12_377::affine_exed_64bit &bases_result,
        CURVE_BLS12_377::affine_exed_64bit &bases_origin)
    {
        bases_result = bases_origin;

#if EXED_UNIFIED_PADD
#if EXED_UNIFIED_PADD_PRECOMPUTE
        bases_result.t = bases_origin.t.get_field_EXED_ADD_K() * bases_origin.t;
#else
        // TODO
#endif
#else
#if EXED_NONE_UNIFIED_PADD_PRECOMPUTE
        bases_result.t = bases_origin.t + bases_origin.t;
        bases_result.x = bases_origin.y + bases_origin.x;
        bases_result.y = bases_origin.y - bases_origin.x;
        // TODO
#endif
#endif
    }

    void precompute_exed_bases_points_for_64bit(
        CURVE_BLS12_377::affine_exed_64bit *bases_result,
        CURVE_BLS12_377::affine_exed_64bit *bases_origin,
        int count)
    {
	    #pragma omp parallel for
        for (int i = 0; i < count; i++)
        {
            precompute_exed_point_for_64bit(bases_result[i], bases_origin[i]);
        }
    }

}
