#include "curve.h"

// 64bit
affine_sw_64bit convert_exed_affine_to_sw_affine_64bit(affine_exed_64bit &in)
{
    affine_sw_64bit res;
    // if in.y == R1 or in.x == 0
    if (fp_64bit::is_one(in.y) || fp_64bit::is_zero(in.x))
    {
        res.x = fp_64bit::get_field_zero();
        res.y = fp_64bit::get_field_zero();

        return res;
    }

    fp_64bit A = fp_64bit::get_field_one() + in.y;
    fp_64bit B = fp_64bit::get_field_one() - in.y;
    fp_64bit B_inv = fp_64bit::inverse(B);
    fp_64bit mont_x = A * B_inv;

    fp_64bit C = in.x * in.y;
    fp_64bit D = in.x - C;
    fp_64bit D_inv = fp_64bit::inverse(D);
    fp_64bit mont_y = A * D_inv;

    fp_64bit E = mont_x * fp_64bit::get_field_EXED_BETA();

    res.x = E + fp_64bit::get_field_EXED_ALPHA();
    res.y = mont_y *  fp_64bit::get_field_EXED_BETA();

    return res;
}

affine_exed_64bit convert_sw_affine_to_exed_affine_64bit(affine_sw_64bit &in)
{
    affine_exed_64bit res;
    fp_64bit alpha = fp_64bit::get_field_EXED_ALPHA();
    fp_64bit beta = fp_64bit::get_field_EXED_BETA();

    if (fp_64bit::is_zero(in.x) && fp_64bit::is_zero(in.y))
    {
        res.x = fp_64bit::get_field_zero();
        res.y = fp_64bit::get_field_one();
        res.t = fp_64bit::get_field_zero();
        return res;
    }

    fp_64bit beta_inverse = fp_64bit::inverse(beta);
    fp_64bit mont_x = (in.x - alpha);
    mont_x = mont_x * beta_inverse;

    fp_64bit mont_y = in.y * beta_inverse;

    fp_64bit mont_x_one = mont_x + fp_64bit::get_field_one();

    if (fp_64bit::is_zero(mont_y) && fp_64bit::is_zero(mont_x_one))
    {
        res.x = fp_64bit::get_field_zero();
        res.y = fp_64bit::get_field_one();
        res.t = fp_64bit::get_field_zero();
        return res;
    }

    fp_64bit mont_y_inverse = fp_64bit::inverse(mont_y);
    res.x = mont_x * mont_y_inverse;

    res.y = mont_x - fp_64bit::get_field_one();
    fp_64bit mont_x_one_inverse = fp_64bit::inverse(mont_x_one);
    res.y = res.y * mont_x_one_inverse;

    res.t = res.x * res.y;

    res = affine_exed_64bit::to_neg_one_a(res);

    return res;
}