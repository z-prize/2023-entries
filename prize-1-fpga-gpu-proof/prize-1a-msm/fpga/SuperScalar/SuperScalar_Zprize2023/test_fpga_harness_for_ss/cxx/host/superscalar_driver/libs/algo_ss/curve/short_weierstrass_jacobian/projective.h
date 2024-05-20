#ifndef __CURVE_SW_PROJECTIVE__
#define __CURVE_SW_PROJECTIVE__

#include <stdio.h>
#include "field.h"
#include "affine.h"
#include "configuration.h"

namespace CURVE_SW
{

    template <typename Field, unsigned B_VALUE>
    class Projective
    {
        friend Affine<Field, B_VALUE>;

        typedef typename Field::storage storage;
        typedef typename Field::storage_wide storage_wide;

    public:
        Field x;
        Field y;
        Field z;

        static constexpr inline Projective point_at_infinity()
        {
            Field x = Field::get_field_zero();
            Field y = Field::get_field_one();
            Field z = Field::get_field_zero();

            return {x, y, z};
        };

        static inline Projective to_montgomery(const Projective &point)
        {
            Field x = Field::to_montgomery(point.x);
            Field y = Field::to_montgomery(point.y);
            Field z = Field::to_montgomery(point.z);

            return {x, y, z};
        }

        static inline Projective from_montgomery(const Projective &point)
        {
            Field x = Field::from_montgomery(point.x);
            Field y = Field::from_montgomery(point.y);
            Field z = Field::from_montgomery(point.z);

            return {x, y, z};
        }

        static inline Projective neg(const Projective &point)
        {
            Field y = Field::neg(point.y);

            return {point.x, y, point.z};
        }

        static inline Projective init(Projective &p)
        {
            p.x = Field::get_field_zero();
            p.y = Field::get_field_one();
            p.z = Field::get_field_zero();

            return p;
        }


        static inline Affine<Field, B_VALUE> to_affine(const Projective &point)
        {
            if (is_zero(point))
            {
                // Affine::zero()
                Field x = Field::get_field_zero();
                Field y = Field::get_field_one();
                return {x, y};
            }
            else if (is_one(point))
            {
                return {point.x, point.y};
            }
            else
            {
                // TODO
                Field zinv{};

                zinv = Field::inverse(point.z);

                Field zinv_squared = zinv ^ 2;
                Field x = point.x * zinv_squared;
                Field tmp = zinv * zinv_squared;
                Field y = point.y * tmp;

                return {x, y};
            }

            return {point.x, point.y};
        }

        static /* inline */ bool is_zero(const Projective &point)
        {
            return Field::is_zero(point.z);
        }

        static inline bool is_one(const Projective &point)
        {
            return Field::is_one(point.z);
        }

        static inline bool is_unity_one(const Projective &point)
        {
            return Field::is_unity_one(point.z);
        }

        static /* inline */ Projective dbl(const Projective &point)
        {
            return dbl_2009_l(point);
        }

        // Add  F2
        friend /* inline */ Projective operator+(Projective p1, const Projective &p2)
        {
            return add_2007_bl(p1, p2);
        }

        // Madd  F3
        friend /* inline */ Projective operator+(Projective p1, const Affine<Field, B_VALUE> &p2)
        {
            return madd_2007_bl(p1, p2);
        }

        // Add  F2
        static constexpr inline Projective add(Projective p1, const Projective &p2, CONFIGURATION *configuration)
        {
            return add_2007_bl(p1, p2, configuration);
        }

        // Madd  F3
        static constexpr inline Projective madd(Projective p1, const Affine<Field, B_VALUE> &p2, CONFIGURATION *configuration)
        {
            return madd_2007_bl(p1, p2, configuration);
        }


        static constexpr inline void print(const Projective &point)
        {
            printf("  SW Projective {\n");
            printf("    x:");
            Field::print(point.x);
            printf("    y:");
            Field::print(point.y);
            printf("    z:");
            Field::print(point.z);
            printf("  }\n");
        }

        static constexpr inline void print(const char *info, const Projective &point)
        {
            printf("  %s\n", info);
            printf("  SW Projective {\n");
            printf("    x:");
            Field::print(point.x);
            printf("    y:");
            Field::print(point.y);
            printf("    z:");
            Field::print(point.z);
            printf("  }\n");
        }

        static inline Projective dbl_2009_l(const Projective &point)
        {
            Projective res = {};

            if (Projective::is_zero(point))
            {
                res = point;
            }

            res.z = point.y * point.z;
            res.z = res.z + res.z;

            Field a = point.x ^ 2;
            Field b = point.y ^ 2;
            Field c = b ^ 2;
            Field d = point.x + b;
            d = d ^ 2;
            d = d - a;
            d = d - c;
            d = d + d;
            Field e = a + a;
            e = e + a;
            Field f = e ^ 2;

            res.x = d + d;
            res.x = f - res.x;
            res.y = d - res.x;
            res.y = res.y * e;
            Field c3 = c + c;
            c3 = c3 + c3;
            c3 = c3 + c3;
            res.y = res.y - c3;

            return res;
        }

        static inline Projective add_2007_bl(const Projective &p1, const Projective &p2, CONFIGURATION *configuration = nullptr)
        {
            Projective res = {};
            if (Projective::is_zero(p2))
            {
                res = p1;
                return res;
            }

            if (Projective::is_zero(p1))
            {
                res = p2;
                return res;
            }

            Field z1z1 = p1.z ^ 2;
            Field z2z2 = p2.z ^ 2;
            Field u1 = p1.x * z2z2;
            Field u2 = p2.x * z1z1;
            Field s1 = p1.y * p2.z;
            s1 = s1 * z2z2;
            Field s2 = p2.y * p1.z;
            s2 = s2 * z1z1;

            if ((u1 == u2) && (s1 == s2))
            {
                return dbl(p1);
            }

            Field h = u2 - u1;
            Field hh = h ^ 2;
            Field i = hh + hh;
            i = i + i;
            Field j = h * i;
            Field r = s2 - s1;
            r = r + r;
            Field v = u1 * i;

            res.x = r ^ 2;
            res.x = res.x - j;
            res.x = res.x - v;
            res.x = res.x - v;

            j = s1 * j;
            j = j + j;
            res.y = v - res.x;
            res.y = res.y * r;
            res.y = res.y - j;

            res.z = p1.z + p2.z;
            res.z = res.z ^ 2;
            res.z = res.z - z1z1;
            res.z = res.z - z2z2;
            res.z = res.z * h;

            return res;
        }

        static inline Projective madd_2007_bl(const Projective &p1, const Affine<Field, B_VALUE> &p2, CONFIGURATION *configuration = nullptr)
        {
            if (configuration)
            {
                printf("    madd_2007_bl F3\n");
            }

            Projective res = {};

            if (Affine<Field, B_VALUE>::is_zero(p2))
            {
                res = p1;
                return res;
            }

            if (Projective::is_zero(p1))
            {
                res.x = p2.x;
                res.y = p2.y;
                res.z = Field::get_field_one();
                return res;
            }

            Field z1z1 = p1.z ^ 2;

            Field u2 = p2.x * z1z1;
            Field s2 = p2.y * p1.z;
            s2 = s2 * z1z1;

            if ((p1.x == u2) && (p1.y == s2))
            {
                res = Projective::dbl(p1);
                return res;
            }

            Field h = u2 - p1.x;
            Field hh = h ^ 2;
            Field i = hh + hh;
            i = i + i;

            Field j = h * i;
            Field r = s2 - p1.y;
            r = r + r;
            Field v = p1.x * i;
            res.x = r ^ 2;
            res.x = res.x - j;
            res.x = res.x - v;
            res.x = res.x - v;
            j = p1.y * j;
            j = j + j;
            res.y = v - res.x;
            res.y = res.y * r;
            res.y = res.y - j;
            res.z = p1.z + h;
            res.z = res.z ^ 2;
            res.z = res.z - z1z1;
            res.z = res.z - hh;

            return res;
        }
    };

}


#endif