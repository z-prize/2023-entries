#ifndef __CURVE_SW_AFFINE__
#define __CURVE_SW_AFFINE__

#include <stdio.h>

namespace CURVE_SW
{

    template <typename Field, unsigned B_VALUE>
    class Projective;

    template <typename Field, unsigned B_VALUE>
    class Affine
    {
        // friend Projective<Field, B_VALUE>;

        typedef Field field;
        typedef typename Field::storage storage;
        typedef typename Field::storage_wide storage_wide;

    public:
        field x;
        field y;

        static inline Affine to_montgomery(const Affine &point)
        {
            field x = field::to_montgomery(point.x);
            field y = field::to_montgomery(point.y);

            return {x, y};
        }

        static inline Affine from_montgomery(const Affine &point)
        {
            field x = field::from_montgomery(point.x);
            field y = field::from_montgomery(point.y);

            return {x, y};
        }

        static inline Affine neg(const Affine &point)
        {
            field y = field::neg(point.y);

            return {point.x, y};
        }

        static inline Projective<Field, B_VALUE> to_projective(const Affine &point)
        {
            field z = field::get_field_one();

            return Projective<Field, B_VALUE>{point.x, point.y, z};
        }

        static /* inline */ bool is_zero(const Affine &point)
        {
            return field::is_zero(point.x);
        }

        static constexpr inline void print(const Affine &point)
        {
            printf("  SW Affine {\n");
            printf("    x:");
            Field::print(point.x);
            printf("    y:");
            Field::print(point.y);
            printf("  }\n");
        }

        static constexpr inline void print(const char *info, const Affine &point)
        {
            printf("  %s\n", info);
            printf("  SW Affine {\n");
            printf("    x:");
            Field::print(point.x);
            printf("    y:");
            Field::print(point.y);
            printf("  }\n");
        }
    };

}



#endif