#ifndef __CURVE_H__
#define __CURVE_H__

#include "field.h"
#include "bls12_377_config.h"

using namespace CURVE_BLS12_377;

#define SW_AFFINE_BYTES 96

affine_sw_64bit convert_exed_affine_to_sw_affine_64bit(affine_exed_64bit &in);

affine_exed_64bit convert_sw_affine_to_exed_affine_64bit(affine_sw_64bit &in);

#endif