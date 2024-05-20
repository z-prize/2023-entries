#ifndef __CURVE_EXED_AFFINE_H__
#define __CURVE_EXED_AFFINE_H__

#include <stdio.h>

namespace CURVE_EXED
{

    template <typename Field, unsigned B_VALUE>
    class Projective;

    template <typename Field, unsigned B_VALUE>
    class Affine
    {
        typedef Field field;
        typedef typename Field::storage storage;
        typedef typename Field::storage_wide storage_wide;

    public:
        field x;
        field y;
        field t;

        static inline Affine to_montgomery(const Affine &point)
        {
            field x = field::to_montgomery(point.x);
            field y = field::to_montgomery(point.y);
            field t = field::to_montgomery(point.t);
            return {x, y, t};
        }

        static inline Affine from_montgomery(const Affine &point)
        {
            field x = field::from_montgomery(point.x);
            field y = field::from_montgomery(point.y);
            field t = field::from_montgomery(point.t);

            return {x, y, t};
        }

        static inline Affine neg(const Affine &point)
        {
            field x = field::neg(point.x);
            field t = field::neg(point.t);
    
            return {x, point.y, t};
        }

        static inline Affine from_neg_one_a(const Affine &point)
        {
            Field x = point.x * Field::get_field_EXED_COEFF_SQRT_NEG_A_INV();
            Field t = point.y * x;

            return {x, point.y, t};
        }

        static inline Affine to_neg_one_a(const Affine &point)
        {
            Field x = point.x * Field::get_field_EXED_COEFF_SQRT_NEG_A();
            Field t = point.y * x;

            return {x, point.y, t};
        }

        static /* inline */ bool is_zero(const Affine &point)
        {
            return field::is_zero(point.x);
        }

        static constexpr inline void print(const Affine &point)
        {
            printf("  EXED Affine {\n");
            printf("    x:");
            Field::print(point.x);
            printf("    y:");
            Field::print(point.y);
            printf("    t:");
            Field::print(point.t);
            printf("  }\n");
        }

        static constexpr inline void print(const char *info, const Affine &point)
        {
            printf("  %s\n", info);
            printf("  EXED Affine {\n");
            printf("    x:");
            Field::print(point.x);
            printf("    y:");
            Field::print(point.y);
            printf("    t:");
            Field::print(point.t);
            printf("  }\n");
        }
    };

}


#endif