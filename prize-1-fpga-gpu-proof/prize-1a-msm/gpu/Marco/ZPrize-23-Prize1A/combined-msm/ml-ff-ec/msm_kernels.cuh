#pragma once
#include "ec.cuh"
#include "ff_dispatch_st.cuh"

namespace msm {

#if 0
    typedef ec<fd381_p> curve;
#else
    typedef ec<fd377_p> curve;
#endif

typedef curve::field field;
typedef curve::point_affine point_affine;
typedef curve::point_xyzz point_xyzz;

inline __device__
point_xyzz  curve_add(const point_xyzz & p0, const point_affine & p1, bool is381) {
    if (is381) {
        auto ret = ec<fd381_p>::add(*(ec<fd381_p>::point_xyzz*)&p0, *(ec<fd381_p>::point_affine*)&p1, ec<fd381_p>::field());
        return *(point_xyzz*)&ret;
    }

    auto ret = ec<fd377_p>::add(*(ec<fd377_p>::point_xyzz*)&p0, *(ec<fd377_p>::point_affine*)&p1, ec<fd377_p>::field());
    return *(point_xyzz*)&ret;
}

inline __device__
point_xyzz  curve_add(const point_xyzz & p0, const point_xyzz & p1, bool is381) {
    if (is381) {
        auto ret = ec<fd381_p>::add(*(ec<fd381_p>::point_xyzz*)&p0, *(ec<fd381_p>::point_xyzz*)&p1, ec<fd381_p>::field());
        return *(point_xyzz*)&ret;
    }
    auto ret = ec<fd377_p>::add(*(ec<fd377_p>::point_xyzz*)&p0, *(ec<fd377_p>::point_xyzz*)&p1, ec<fd377_p>::field());
    return *(point_xyzz*)&ret;
}

inline __device__
point_xyzz  curve_dbl(const point_xyzz & p0, bool is381) {
    if (is381) {
        auto ret = ec<fd381_p>::dbl(*(ec<fd381_p>::point_xyzz*)&p0, ec<fd381_p>::field());
        return *(point_xyzz*)&ret;
    }
    auto ret = ec<fd377_p>::dbl(*(ec<fd377_p>::point_xyzz*)&p0, ec<fd377_p>::field());
    return *(point_xyzz*)&ret;
}

inline __device__
point_xyzz point_at_infinity(bool is381) {
    if (is381) {
        auto ret = ec<fd381_p>::point_xyzz::point_at_infinity(ec<fd381_p>::field());
        return *(point_xyzz*)&ret;
    }
    auto ret = ec<fd377_p>::point_xyzz::point_at_infinity(ec<fd377_p>::field());
    return *(point_xyzz*)&ret;
}

inline __device__  point_affine to_affine(const point_xyzz & point, bool is381) {
    if (is381) {
        auto ret = ((ec<fd381_p>::point_xyzz*)&point)->to_affine();
        return *(point_affine *)&ret;
    }
    auto ret = ((ec<fd377_p>::point_xyzz*)&point)->to_affine();
    return *(point_affine *)&ret;
}

inline __device__  point_affine point_neg(const point_affine &point, bool is381) {
    if (is381) {
        auto ret = ec<fd381_p>::point_affine::neg(*(ec<fd381_p>::point_affine*)&point, ec<fd381_p>::field());
        return *(point_affine *)&ret;
    }
    auto ret = ec<fd377_p>::point_affine::neg(*(ec<fd377_p>::point_affine*)&point, ec<fd377_p>::field());
    return *(point_affine *)&ret;
}

} // namespace msm