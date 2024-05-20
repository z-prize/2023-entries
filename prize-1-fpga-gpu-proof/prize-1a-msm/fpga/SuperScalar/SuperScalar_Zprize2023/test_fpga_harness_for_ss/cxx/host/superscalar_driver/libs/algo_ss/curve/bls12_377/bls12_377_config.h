#ifndef __CURVE_BLS12_377_CONFIG_H__
#define __CURVE_BLS12_377_CONFIG_H__

#include "paramter_64bit.h"
#include "short_weierstrass_jacobian/affine.h"
#include "short_weierstrass_jacobian/projective.h"
#include "twisted_edwards_extended/affine.h"
#include "twisted_edwards_extended/projective.h"
#include "field.h"

namespace CURVE_BLS12_377
{
    const int FP_BIT_WIDTH = 377;
    const int FR_BIT_WIDTH = 253;

    // radix 64bit
    typedef Field<uint64_t, fp_configuration_64bit> fp_64bit;// Fq
    typedef Field<uint64_t, fr_configuration_64bit> fr_64bit;// Fr

    // SW
    typedef CURVE_SW::Projective<fp_64bit, WEIERSTRASS_B> projective_sw_64bit;
    typedef CURVE_SW::Affine<fp_64bit, WEIERSTRASS_B> affine_sw_64bit;
    // EXED
    typedef CURVE_EXED::Projective<fp_64bit, WEIERSTRASS_B> projective_exed_64bit;
    typedef CURVE_EXED::Affine<fp_64bit, WEIERSTRASS_B> affine_exed_64bit;

} // namespace  CURVE_BLS12_377


#endif