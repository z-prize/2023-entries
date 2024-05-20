#ifndef __MATH_64IT_H__
#define __MATH_64IT_H__

#include <stdio.h>
#include <cstdint>
#include <math.h>
#include <inttypes.h>

#include "math.h"

namespace MATH_64BIT
{

    // #define UINT64_BORROW ((uint128_t)pow(2,64))
    #define UINT64_BORROW ((uint128_t)(1 << 64))
    
    inline uint64_t add_with_carry_in(uint64_t a, uint64_t b, uint64_t *carry)
    {
        uint128_t tmp = (uint128_t)a + (uint128_t)b + (uint128_t)*carry;

        *carry = (uint64_t)(tmp >> 64);

        return (uint64_t)tmp;
    }

    inline uint64_t add_carry_with_carry(uint64_t a, uint64_t *carry)
    {
        uint128_t tmp = (uint128_t)a  + (uint128_t)*carry;

        *carry = (uint64_t)(tmp >> 64);

        return (uint64_t)tmp;
    }

    inline uint64_t sub(uint64_t a, uint64_t b)
    {
        uint128_t c = (uint128_t)a - (uint128_t)b;

        return (uint64_t)c;
    }

    inline uint64_t sub_with_borrow(uint64_t a, uint64_t b, uint64_t *borrow)
    {
        uint64_t tmp = 0x0;
        uint64_t c = 0x0;
        uint64_t carry_in = *borrow;
        uint64_t carry_out = 0;

        /* case: 0 - 1  */
        if ((0x0 == a) && (1 == carry_in))
        {
            tmp = (uint64_t)(UINT64_BORROW - 1);
            carry_out = 1;
        }
        else    /* a >= tmp */
        {
            tmp = a - carry_in;
        }

        if ( tmp >= b)
        {
            c = sub(tmp, b);
        }
        else
        {
            c = (uint64_t)((uint128_t)tmp + UINT64_BORROW - (uint128_t)b);
            carry_out = 1;
        }

        *borrow = carry_out;
        return c;
    }

    inline uint64_t mul_with_carry(uint64_t a, uint64_t b, uint64_t *carry)
    {
        uint128_t tmp = (uint128_t)a * (uint128_t)b;

        *carry = (uint64_t)(tmp >> 64);

        return (uint64_t)tmp;
    }

    inline uint64_t mac_with_carry_in(uint64_t a, uint64_t b, uint64_t c, uint64_t *carry)
    {

        uint128_t tmp = (uint128_t)a * (uint128_t)b + (uint128_t)c + (uint128_t)*carry;

        *carry = (uint64_t)(tmp >> 64);

        return (uint64_t)tmp;
    }



}


#endif