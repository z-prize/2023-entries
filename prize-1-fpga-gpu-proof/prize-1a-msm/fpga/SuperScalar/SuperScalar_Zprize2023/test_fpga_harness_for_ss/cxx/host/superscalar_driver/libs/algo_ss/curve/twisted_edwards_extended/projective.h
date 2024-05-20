
#ifndef __CURVE_EXED_PROJECTIVE_H__
#define __CURVE_EXED_PROJECTIVE_H__

#include <stdio.h>
#include "field.h"
#include "affine.h"
#include "config.h"

namespace CURVE_EXED
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
        Field t;
        Field z;

        static inline Projective to_montgomery(const Projective &point)
        {
            Field x = Field::to_montgomery(point.x);
            Field y = Field::to_montgomery(point.y);
            Field t = Field::to_montgomery(point.t);
            Field z = Field::to_montgomery(point.z);

            return {x, y, t, z};
        }

        static inline Projective from_montgomery(const Projective &point)
        {
            Field x = Field::from_montgomery(point.x);
            Field y = Field::from_montgomery(point.y);
            Field t = Field::from_montgomery(point.t);
            Field z = Field::from_montgomery(point.z);

            return {x, y, t, z};
        }

        static inline Projective init(Projective &p)
        {
            p.x = Field::get_field_zero();
            p.y = Field::get_field_one();
            p.t = Field::get_field_zero();
            p.z = Field::get_field_one();

            return p;
        }

        static inline Affine<Field, B_VALUE> to_affine(const Projective &point)
        {
            if (is_zero(point))
            {
                // Affine::zero()
                Field x = Field::get_field_zero();
                Field y = Field::get_field_one();
                Field t = Field::get_field_zero();
                return {x, y, t};
            }

            // TODO
            Field zinv = Field::inverse(point.z);
            Field x = point.x * zinv;
            Field y = point.y * zinv;
            Field t = x * y;

            return {x, y, t};
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

        static constexpr inline void print(const Projective &point)
        {
            printf("  EXED Projective {\n");
            printf("    x:");
            Field::print(point.x);
            printf("    y:");
            Field::print(point.y);
            printf("    t:");
            Field::print(point.t);
            printf("    z:");
            Field::print(point.z);
            printf("  }\n");
        }

        static constexpr inline void print(const char *info, const Projective &point)
        {
            printf("  %s\n", info);
            printf("  EXED Projective {\n");
            printf("    x:");
            Field::print(point.x);
            printf("    y:");
            Field::print(point.y);
            printf("    t:");
            Field::print(point.t);
            printf("    z:");
            Field::print(point.z);
            printf("  }\n");
        }

        /************************************************************************************************ 
        - Function :        dbl_2009_l
        - Description :     EXED PDBL Projective function
                            dbl-2009-l
                            http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
                            Cost: 2M + 5S + 6add + 3*2 + 1*3 + 1*8.
                            Source: 2009.04.01 Lange.
                            Explicit formulas:
                                A = X1^2
                                B = Y1^2
                                C = B^2
                                D = 2*((X1+B)^2-A-C)
                                E = 3*A
                                F = E^2
                                X3 = F-2*D
                                Y3 = E*(D-X3)-8*C
                                Z3 = 2*Y1*Z1

                            Note: Z1 = 1
        - Input params:
                1. const blst_p1_exed_27bit* p                  :  pointer of p(EXED Projective)
                                                                    (x1,y1,t1,z1)
        - Output params:
                1. blst_p1_exed_27bit *out                      :  pointer of out(EXED Projective)
                                                                    (x3,y3,t3,z3)    
        - Return:           void
        - Others:
        **************************************************************************************************/
        static inline Projective dbl_2009_l(const Projective &point)
        {
            Projective res = {};

            Field A = point.x ^ 2;
            Field B = point.y ^ 2;
            Field C1 = point.z ^ 2;
            Field C = C1 + C1;

            Field D = Field::get_field_zero() - A;

            Field E1 = point.x + point.y;
            Field E2 = E1 ^ 2;
            Field E3 = E2 - A;
            Field E = E3 - B;

            Field G = D + B;
            Field F = G + C;
            Field H = D - B;

            res.x = E * F;
            res.y = G * H;
            res.t = E * H;
            res.z = F * G;

            return res;
        }

        // Add
        friend /* inline */ Projective operator+(Projective p1, const Projective &p2)
        {
            #if EXED_UNIFIED_PADD
            return add_2008_hwcd_3_unified_padd(p1, p2);
            #else
            return add_2008_hwcd_3(p1, p2);
            #endif
        }

        // Madd
        friend /* inline */ Projective operator+(Projective p1, const Affine<Field, B_VALUE> &p2)
        {
            #if EXED_UNIFIED_PADD
            return madd_2008_hwcd_3(p1, p2);
            #else
            return madd_2008_hwcd_4(p1, p2);
            #endif
        }

        // Add
        static constexpr inline Projective add(Projective p1, const Projective &p2, CONFIGURATION *configuration = nullptr)
        {
            #if EXED_UNIFIED_PADD
            return add_2008_hwcd_3_unified_padd(p1, p2, configuration);
            #else
            return add_2008_hwcd_3(p1, p2, configuration);
            #endif
        }

        // Madd
        static constexpr inline Projective madd(Projective p1, const Affine<Field, B_VALUE> &p2, CONFIGURATION *configuration = nullptr)
        {
            #if EXED_UNIFIED_PADD
            return madd_2008_hwcd_3(p1, p2, configuration);
            #else
            return madd_2008_hwcd_4(p1, p2, configuration);
            #endif
        }

        #if EXED_UNIFIED_PADD
        /************************************************************************************************ 
        - Function :        add_2008_hwcd_3_unified_padd
        - Description :     EXED PADD F2 function
                            
                            http://eprint.iacr.org/2008/522  HWCD08, Sec. 4.2
                            or
                            https://vendiblelabs.com/documents/sovereign_system_v1.1.pdf 
                            
                            https://www.hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html  

                            unified padd: 
                            A = (y1 - x1) * (y2 - x2)   // R5 = R1 * R2
                            B = (y1 + x1) * (y2 + x2)   // R6 = R3 * R4
                            C = t1 * k * t2   // R7 = t1 * t2   R7 = k*R7
                            D = z1 * 2 * z2   // R8 = z1 * z2   R8 = R8 *2
                            E = B - A         // R1 = R6 - R5
                            F = D - C         // R2 = R8 - R7 
                            G = D + C         // R3 = R8 + R7
                            H = B + A         // R4 = R6 + R5
                            X3 = E * F        // X3 = R1 * R2
                            Y3 = G * H        // Y3 = R3 * R4
                            T3 = E * H        // T3 = R1 * R4
                            Z3 = F * G        // Z3 = R2 * R3

    
                            t2 = k * t2
                            z1 = z1 * z2
                            PADD_F3( (x1, y1, t1, z1), (x2, y2, t2))

        - Input params:
                1. Projective &p1                   :  p1 (x1,y1,t1,z1)
                2. const Projective &p2             :  p2 (x2,y2,t2,z2)
        - Output params:

        - Return:
                1. Projective                       :  Projective point(x3,y3,t3,z3)
        - Others:
                F2 padd
        **************************************************************************************************/
        // F2
        static inline Projective add_2008_hwcd_3_unified_padd(Projective &p1, const Projective &p2, CONFIGURATION *configuration = nullptr)
        {

            Affine<Field, B_VALUE> p2_affine;
            p2_affine.x = p2.x;
            p2_affine.y = p2.y;

            p2_affine.t = Field::get_field_EXED_ADD_K() * p2.t;
            p1.z = p1.z * p2.z;

            return madd_2008_hwcd_3(p1, p2_affine);
        }

        /************************************************************************************************ 
        - Function :        madd_2008_hwcd_3
        - Description :     EXED PADD F3 function
                            
                            http://eprint.iacr.org/2008/522  HWCD08, Sec. 4.2
                            or
                            https://vendiblelabs.com/documents/sovereign_system_v1.1.pdf 
                            
                            unified padd: 
                            A = (y1 - x1) * (y2 - x2)   // R5 = R1 * R2
                            B = (y1 + x1) * (y2 + x2)   // R6 = R3 * R4
                            C = t1 * k * t2   // R7 = t1 * t2   R7 = k*R7
                            D = z1 * 2 * z2   // R8 = z1 * z2   R8 = R8 *2
                            E = B - A         // R1 = R6 - R5
                            F = D - C         // R2 = R8 - R7 
                            G = D + C         // R3 = R8 + R7
                            H = B + A         // R4 = R6 + R5
                            X3 = E * F        // X3 = R1 * R2
                            Y3 = G * H        // Y3 = R3 * R4
                            T3 = E * H        // T3 = R1 * R4
                            Z3 = F * G        // Z3 = R2 * R3

    
                            After simplification:  z2 = 1
                            A = (y1 - x1) * (y2 - x2)
                            B = (y1 + x1) * (y2 + x2)
                            C = t1 * t2   // t2 = k * t2   precomute
                            D = z1 + z1   // D = z1 * 2 = z1 + z1
                            E = B - A
                            F = D - C
                            G = D + C
                            H = B + A
                            X3 = E * F
                            Y3 = G * H
                            T3 = E * H
                            Z3 = F * G


        - Input params:
                1. const Projective &p1                 :  p1 (x1,y1,t1,z1)
                2. const Affine<Field, B_VALUE> &p2     :  p2 (x2,y2,t2,z2)
        - Output params:

        - Return:
                1. Projective                           :  Projective point(x3,y3,t3,z3)
        - Others:
        **************************************************************************************************/
        static inline Projective madd_2008_hwcd_3(const Projective &p1, const Affine<Field, B_VALUE> &p2, CONFIGURATION *configuration = nullptr)
        {
            Projective res = {};

            Field A1 = p1.y - p1.x;
            Field A2 = p2.y - p2.x;
            Field A = A1 * A2;

            Field B1 = p1.y + p1.x;
            Field B2 = p2.y + p2.x;
            Field B = B1 * B2;

            #if EXED_UNIFIED_PADD_PRECOMPUTE
            Field C = p1.t * p2.t;
            #else
            Field C1 = Field::get_field_EXED_ADD_K() * p1.t;
            Field C = C1 * p2.t;
            #endif

            Field D = p1.z + p1.z;
            Field E = B - A;
            Field F = D - C;
            Field G = D + C;
            Field H = B + A;

            res.x = E * F;
            res.y = G * H;
            res.t = E * H;
            res.z = F * G;

            return res;
        }
        #else   //EXED_UNIFIED_PADD
        /************************************************************************************************ 
        - Function :        add_2008_hwcd_3
        - Description :     EXED PADD F2 function

                            https://www.hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html add-2008-hwcd-3

                            The "add-2008-hwcd-3" addition formulas:
                            Assumptions: k=2*d.
                            Cost: 8M + 1*k + 8add + 1*2.
                            Cost: 8M + 6add dependent upon the first point.
                            Source: 2008 Hisil–Wong–Carter–Dawson, http://eprint.iacr.org/2008/522, Section 3.1.
                            Strongly unified.
                            Explicit formulas:
                                A = (Y1-X1)*(Y2-X2)
                                B = (Y1+X1)*(Y2+X2)
                                C = T1*k*T2
                                D = Z1*2*Z2
                                E = B-A
                                F = D-C
                                G = D+C
                                H = B+A
                                X3 = E*F
                                Y3 = G*H
                                T3 = E*H
                                Z3 = F*G

        - Input params:
                1. Projective &p1                   :  p1 (x1,y1,t1,z1)
                2. const Projective &p2             :  p2 (x2,y2,t2,z2)
        - Output params:

        - Return:
                1. Projective                       :  Projective point(x3,y3,t3,z3)
        - Others:
        **************************************************************************************************/
        static inline Projective add_2008_hwcd_3(Projective &p1, const Projective &p2, CONFIGURATION *configuration = nullptr)
        {

            Projective res = {};

            Field A1 = p1.y - p1.x;
            Field A2 = p2.y - p2.x;
            Field A = A1 * A2;

            Field B1 = p1.y + p1.x;
            Field B2 = p2.y + p2.x;
            Field B = B1 * B2;

            Field C1 = Field::get_field_EXED_ADD_K() * p1.t;
            Field C = C1 * p2.t;

            Field D = p1.z * p2.z;
            D = D + D;
            Field E = B - A;
            Field F = D - C;
            Field G = D + C;
            Field H = B + A;

            res.x = E * F;
            res.y = G * H;
            res.t = E * H;
            res.z = F * G;
            
            return res;
        }

        /************************************************************************************************ 
        - Function :        madd_unified_padd
        - Description :     EXED PADD function
                            
                            https://www.hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html madd-2008-hwcd-4 
                            
                            The "madd-2008-hwcd-4" addition formulas:
                            Assumptions: Z2=1.
                            Cost: 7M + 8add + 2*2.
                            Cost: 7M + 6add + 1*2 dependent upon the first point.
                            Source: 2008 Hisil–Wong–Carter–Dawson, http://eprint.iacr.org/2008/522, Section 3.2.
                            Explicit formulas:
                                A = (Y1-X1)*(Y2+X2)
                                B = (Y1+X1)*(Y2-X2)
                                C = Z1*2*T2
                                D = 2*T1
                                E = D+C
                                F = B-A
                                G = B+A
                                H = D-C
                                X3 = E*F
                                Y3 = G*H
                                T3 = E*H
                                Z3 = F*G

                            Note: Z2 = 1

                            However, we do a little optimization. We put the "2" in "C = Z1*2*T2" to the front in base EXED Affine t2.
                            So, Cost: 7M + 8add + 1*2.
                            Explicit formulas:
                                A = (Y1-X1)*(Y2+X2)
                                B = (Y1+X1)*(Y2-X2)
                                C = Z1*T2
                                D = 2*T1
                                E = D+C
                                F = B-A
                                G = B+A
                                H = D-C
                                X3 = E*F
                                Y3 = G*H
                                T3 = E*H
                                Z3 = F*G


        - Input params:
                1. const Projective &p1                 :  p1 (x1,y1,t1,z1)
                2. const Affine<Field, B_VALUE> &p2     :  p2 (x2,y2,t2,z2)
        - Output params:

        - Return:
                1. Projective                           :  Projective point(x3,y3,t3,z3)
        - Others:
        **************************************************************************************************/
        // static inline Projective madd_2008_hwcd_4(const Projective &p1, const Affine<Field, B_VALUE> &p2, CONFIGURATION *configuration)
        static inline Projective madd_2008_hwcd_4(const Projective &p1, const Affine<Field, B_VALUE> &p2, CONFIGURATION *configuration = nullptr)
        {
            Projective res = {};

            // (y1 - x1) * (y2 + x2)  (y1 + x1) * (y2 - x2)
            #if EXED_NONE_UNIFIED_PADD_PRECOMPUTE
            Field A1 = p1.y - p1.x;
            Field A = A1 * p2.x;

            Field B1 = p1.y + p1.x;
            Field B = B1 * p2.y;
            #else
            Field A1 = p1.y - p1.x;
            Field A2 = p2.y + p2.x;
            Field A = A1 * A2;

            Field B1 = p1.y + p1.x;
            Field B2 = p2.y - p2.x;
            Field B = B1 * B2;
            #endif

            Field F = B - A;

            if (Field::is_zero(F))
            {
                return dbl_2009_l(p1);
            }

            // t2
            #if EXED_NONE_UNIFIED_PADD_PRECOMPUTE
            Field C = p1.z * p2.t;
            #else
            Field t2 = p2.t + p2.t;
            Field C = p1.z * t2;
            #endif

            Field D = p1.t + p1.t;
            Field E = D + C;
            Field G = B + A;
            Field H = D - C;

            res.x = E * F;
            res.y = G * H;
            res.t = E * H;
            res.z = F * G;

            return res;
        }

        #endif
    };

}



#endif