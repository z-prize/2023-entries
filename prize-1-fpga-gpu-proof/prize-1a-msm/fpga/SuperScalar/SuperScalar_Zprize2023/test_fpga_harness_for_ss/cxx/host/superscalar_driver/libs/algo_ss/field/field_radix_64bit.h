#ifndef __FIELD_RADIX_64BIT_H__
#define __FIELD_RADIX_64BIT_H__

#include <cstring>
#include "field_storage.h"
#include "common.h"
#include "math_64bit.h"

using namespace common;

namespace FIELD_RADIX_64BIT
{
    // l >= r is 1
    template <typename limb_t, class storage>
    inline int is_ge_limbs(const storage &left, const storage &right, unsigned LC)
    {
        for (int i = (LC - 1); i >= 0; --i)
        {
            if (left.limbs[i] < right.limbs[i])
            {
                return 0;
            }
            else if (left.limbs[i] > right.limbs[i])
            {
                return 1;
            }
        }
        return 1;
    }

    // l > r is 1
    template <typename limb_t, class storage>
    inline int is_gt_limbs(const storage &left, const storage &right, unsigned LC)
    {
        for (int i = (LC - 1); i >= 0; --i)
        {
            if (left.limbs[i] < right.limbs[i])
            {
                return 0;
            }
            else if (left.limbs[i] > right.limbs[i])
            {
                return 1;
            }
        }
        return 0;
    }

    template <class storage>
    inline int is_equal(const storage &left, const storage &right, unsigned LC)
    {
        for (int i = (LC - 1); i >= 0; --i)
        {
            if (left.limbs[i] != right.limbs[i])
            {
                return 0;
            }
        }

        return 1;
    }

    template <typename limb_t, class storage>
    inline void add_mod_limbs_unchecked(storage &res, const storage &x, const storage &y, unsigned LC)
    {
        limb_t carry = 0x0;

        uint64_t carry_value = static_cast<uint64_t>(carry);

        for (unsigned i = 0; i < LC; i++)
        {
            res.limbs[i] = MATH_64BIT::add_with_carry_in((uint64_t)x.limbs[i], (uint64_t)y.limbs[i], &carry_value);
        }
    }

    template <typename limb_t, class storage>
    static inline void sub_mod_limbs_unchecked(storage &res, const storage &x, const storage &y, unsigned LC)
    {
        limb_t borrow = 0x0;

        uint64_t borrow_value = static_cast<uint64_t>(borrow);

        for (unsigned i = 0; i < LC; i++)
        {
            res.limbs[i] = MATH_64BIT::sub_with_borrow((uint64_t)x.limbs[i], (uint64_t)y.limbs[i], &borrow_value);
        }
    }

    template <typename limb_t, class storage>
    static inline void reduce_limbs(storage &x, const storage &m, unsigned LC)
    {
        if (is_ge_limbs<limb_t, storage>(x, m, LC))
        {
            storage x_sub = {};
            sub_mod_limbs_unchecked<limb_t, storage>(x_sub, x, m, LC);
            memcpy(&x, &x_sub, sizeof(storage));
        }
    }

    template <typename limb_t, class storage>
    static constexpr inline void add_mod_limbs_host(storage &res, const storage &x, const storage &y, const storage &m, unsigned LC)
    {
        add_mod_limbs_unchecked<limb_t, storage>(res, x, y, LC);
        reduce_limbs<limb_t, storage>(res, m, LC);
    }

    template <typename limb_t, class storage>
    static constexpr inline void add_limbs_host(storage &res, const storage &x, const storage &y, unsigned LC)
    {
        add_mod_limbs_unchecked<limb_t, storage>(res, x, y, LC);
    }

    template <typename limb_t, class storage>
    static constexpr inline void sub_mod_limbs_host(storage &res, const storage &x, const storage &y, const storage &m, unsigned LC)
    {
        storage added = x;

        if (is_ge_limbs<limb_t, storage>(y, x, LC))
        {
            add_mod_limbs_unchecked<limb_t, storage>(added, added, m, LC);
        }

        sub_mod_limbs_unchecked<limb_t>(res, added, y, LC);
    }

    template <typename limb_t, class storage>
    static constexpr inline void sub_limbs_host(storage &res, const storage &x, const storage &y, unsigned LC)
    {
        sub_mod_limbs_unchecked<limb_t>(res, x, y, LC);
    }

    template <typename limb_t, class storage, class storage_wide>
    inline void mul_limbs(storage_wide &res, const storage &x, const storage &y, unsigned LC)
    {

        limb_t carry = 0x0;

        uint64_t carry_value = static_cast<uint64_t>(carry);

        for (unsigned i = 0; i < LC; i++)
        {
            carry_value = 0x0;
            for (unsigned j = 0; j < LC; j++)
            {
                if (i == 0 && j == 0)
                {
// first operation in round 1
                    res.limbs[i + j] = MATH_64BIT::mul_with_carry((uint64_t)x.limbs[i + j], (uint64_t)y.limbs[i + j], &carry_value);
                }
                else
                {
                    res.limbs[i + j] = MATH_64BIT::mac_with_carry_in((uint64_t)x.limbs[i], (uint64_t)y.limbs[j], (uint64_t)res.limbs[i + j], &carry_value);
                }
            }
            res.limbs[i + LC] = carry_value;
        }
    }

    template <typename limb_t, class storage, class storage_wide>
    void sqr_limbs(storage_wide &res, const storage &x, unsigned LC)
    {
        limb_t carry = 0x0;

        uint64_t carry_value = static_cast<uint64_t>(carry);

        for (unsigned i = 0; i < LC - 1; i++)
        {
            carry_value = 0x0;
            for (unsigned j = 0; j < LC - 1 - i; j++)
            {
                if (i == 0 && j == 0)
                {
                    res.limbs[1] = MATH_64BIT::mul_with_carry((uint64_t)x.limbs[0], (uint64_t)x.limbs[1], &carry_value);
                }
                else
                {
                    res.limbs[i * 2 + j + 1] = MATH_64BIT::mac_with_carry_in((uint64_t)x.limbs[i], (uint64_t)x.limbs[i + j + 1], (uint64_t)res.limbs[i * 2 + j + 1], &carry_value);
                }
            }
            res.limbs[i + 1 + LC - 1] = carry_value;
        }

        for (unsigned i = 0; i < 2 * LC - 1; i++)
        {
            // note
            unsigned rshift_count = 8 * sizeof(limb_t) - 1;

            if (i == 0)
            {
                res.limbs[2 * LC - 1] = res.limbs[2 * LC - 2] >> rshift_count;
            }
            else if (i == 2 * LC - 2)
            {
                res.limbs[2 * LC - 1 - i] = (res.limbs[2 * LC - 1 - i] << 1);
            }
            else
            {
                res.limbs[2 * LC - 1 - i] = (res.limbs[2 * LC - 1 - i] << 1) | (res.limbs[2 * LC - 2 - i] >> rshift_count);
            }
        }

        carry_value = 0x0;
        for (unsigned i = 0; i < 2 * LC; i++)
        {
            if (i % 2 == 0)
            {
                res.limbs[i] = MATH_64BIT::mac_with_carry_in((uint64_t)x.limbs[i >> 1], (uint64_t)x.limbs[i >> 1], (uint64_t)res.limbs[i], &carry_value);
            }
            else
            {

                res.limbs[i] = MATH_64BIT::add_carry_with_carry((uint64_t)res.limbs[i], &carry_value);
            }
        }
    }

    template <typename limb_t, class storage, class storage_wide>
    inline void mont_limbs(storage &res, storage_wide &r, const storage &m, limb_t m_inv, unsigned LC) //(blst_fp ret.limbs, blst_fp_double r, const blst_fp p, uint64_t p_inv)
    {
        limb_t carry2 = 0;

        uint64_t carry2_value = static_cast<uint64_t>(carry2);

        for (unsigned i = 0; i < LC; i++)
        {
            limb_t k = r.limbs[i] * m_inv;
            limb_t carry1 = 0;

            uint64_t carry1_value = static_cast<uint64_t>(carry1);

            for (unsigned j = 0; j < LC; j++)
            {
                if (j == 0)
                {

                    MATH_64BIT::mac_with_carry_in((uint64_t)k, (uint64_t)m.limbs[j], (uint64_t)r.limbs[i + j], &carry1_value);
                }
                else
                {

                    r.limbs[i + j] = MATH_64BIT::mac_with_carry_in((uint64_t)k, (uint64_t)m.limbs[j], (uint64_t)r.limbs[i + j], &carry1_value);
                }
            }

            r.limbs[i + LC] = MATH_64BIT::add_with_carry_in((uint64_t)r.limbs[i + LC], carry2_value, &carry1_value);

            carry2_value = carry1_value;
        }

        for (unsigned i = 0; i < LC; i++)
        {
            res.limbs[i] = r.limbs[i + LC];
        }
        reduce_limbs<limb_t, storage>(res, m, LC);
    }

    template <typename limb_t, class storage, class storage_wide>
    static constexpr inline void mul_mod_limbs_host(storage &res, const storage &x, const storage &y, const storage &m, limb_t m_inv, unsigned LC)
    {
        storage_wide r = {};

        mul_limbs<limb_t, storage, storage_wide>(r, x, y, LC);

        mont_limbs(res, r, m, m_inv, LC);
    }

    template <typename limb_t, class storage, class storage_wide>
    static constexpr inline void sqr_mod_limbs_host(storage &res, const storage &x, const storage &m, limb_t m_inv, unsigned LC)
    {
        storage_wide r = {};

        sqr_limbs<limb_t, storage, storage_wide>(r, x, LC);

        mont_limbs(res, r, m, m_inv, LC);
    }

    template <typename limb_t, class storage>
    static inline void _rshift(storage &res, const storage &value, unsigned LC, unsigned LB)
    {
        limb_t LB_minus_1 = LB - 1;

        for (uint32_t i = 0; i < LC - 1; i++)
        {
            res.limbs[i] = (value.limbs[i + 1] << LB_minus_1) | (value.limbs[i] >> 1);
        }
        res.limbs[LC - 1] = value.limbs[LC - 1] >> 1;
    }

    template <typename limb_t, class storage>
    static inline void div_by_2_mod(storage &res, const storage &value, unsigned LC, unsigned LB)
    {
        _rshift<limb_t>(res, value, LC, LB);
    }

    template <typename limb_t, class storage>
    static inline storage inverse(storage &u, storage &v, storage &b, storage &c, const storage &m, limb_t m_inv, const storage one, unsigned LC, unsigned LB)
    {
        while (!(is_equal(u, one, LC)) && !(is_equal(v, one, LC)))
        {
            while ((u.limbs[0] & 0x1) == 0)
            {
                div_by_2_mod<limb_t>(u, u, LC, LB);

                if ((b.limbs[0] & 0x1) != 0)
                {
                    add_limbs_host<limb_t, storage>(b, b, m, LC);
                }

                div_by_2_mod<limb_t>(b, b, LC, LB);
            }

            while ((v.limbs[0] & 0x1) == 0)
            {
                div_by_2_mod<limb_t>(v, v, LC, LB);

                if ((c.limbs[0] & 0x1) != 0)
                {
                    add_limbs_host<limb_t, storage>(c, c, m, LC);
                }
                div_by_2_mod<limb_t>(c, c, LC, LB);
            }

            if (is_gt_limbs<limb_t, storage>(u, v, LC))
            {
                sub_limbs_host<limb_t, storage>(u, u, v, LC);

                sub_mod_limbs_host<limb_t, storage>(b, b, c, m, LC);
            }
            else
            {
                sub_limbs_host<limb_t, storage>(v, v, u, LC);

                sub_mod_limbs_host<limb_t, storage>(c, c, b, m, LC);
            }
        }

        if (is_equal(u, one, LC))
        {
            return b;
        }
        else
        {
            return c;
        }
    }
}

#endif