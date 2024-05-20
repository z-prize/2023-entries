#ifndef __FIELD_H__
#define __FIELD_H__

#include <stdio.h>
#include "field_storage.h"
#include "field_radix_64bit.h"
#include "common.h"
#include "configuration.h"

using namespace common;

template <typename limb_t, class CONFIG>
struct Field
{
    static constexpr unsigned LC = CONFIG::limbs_count;
    static constexpr unsigned RADIX_TYPE = CONFIG::radix_type;
    static constexpr unsigned BC = CONFIG::bits_count;
    static constexpr unsigned LB = CONFIG::limbs_bits;
    static constexpr unsigned BYC = CONFIG::bytes_count;

    typedef field_storage<limb_t, LC> storage;
    typedef field_storage_wide<limb_t, LC> storage_wide;
    storage limbs_storage;

    static constexpr inline Field get_field_zero()
    {
        return Field{CONFIG::ZERO};
    }

    // return one in montgomery form
    static constexpr /* inline */ Field get_field_one()
    {
        return Field{CONFIG::ONE};
    }

    static constexpr inline storage get_one()
    {
        return CONFIG::ONE;
    }

    static constexpr inline storage get_unity_one()
    {
        return CONFIG::UNITY_ONE;
    }

    static constexpr inline storage get_R2()
    {
        return CONFIG::R2;
    }

    static inline Field get_field_R2()
    {
        return Field{CONFIG::R2};
    }

    static inline storage get_modular()
    {
        return CONFIG::MODULAR;
    }

    static constexpr inline Field get_field_modulus()
    {
        return Field{CONFIG::MODULAR};
    }

    static constexpr inline limb_t get_modular_inv()
    {
        return CONFIG::MODULAR_INV;
    }

    static inline Field get_field_EXED_ADD_K()
    {
        return Field{CONFIG::EXED_ADD_K};
    }

    static inline Field get_field_EXED_ALPHA()
    {
        return Field{CONFIG::EXED_ALPHA};
    }

    static inline Field get_field_EXED_BETA()
    {
        return Field{CONFIG::EXED_BETA};
    }

    static inline Field get_field_EXED_COEFF_SQRT_NEG_A()
    {
        return Field{CONFIG::EXED_COEFF_SQRT_NEG_A};
    }

    static inline Field get_field_EXED_COEFF_SQRT_NEG_A_INV()
    {
        return Field{CONFIG::EXED_COEFF_SQRT_NEG_A_INV};
    }

    static inline Field to_montgomery(const Field &x)
    {
        constexpr storage R2 = get_R2();
        const storage MODULAR = get_modular();
        const limb_t MODULAR_INV = get_modular_inv();
        Field res = {};

        switch (RADIX_TYPE)
        {
            case RADIX_TYPE_64BIT:
                FIELD_RADIX_64BIT::mul_mod_limbs_host<limb_t, storage, storage_wide>(res.limbs_storage, x.limbs_storage, R2, MODULAR, MODULAR_INV, LC);
                break;
        }

        return res;
    }

    static inline Field from_montgomery(const Field &x)
    {
        constexpr storage UNITY_ONE = get_unity_one();
        const storage MODULAR = get_modular();
        const limb_t MODULAR_INV = get_modular_inv();
        Field res = {};

        switch (RADIX_TYPE)
        {
            case RADIX_TYPE_64BIT:
                FIELD_RADIX_64BIT::mul_mod_limbs_host<limb_t, storage, storage_wide>(res.limbs_storage, x.limbs_storage, UNITY_ONE, MODULAR, MODULAR_INV, LC);
                break;
        }

        return res;
    }

    static constexpr inline void add_mod_limbs(Field &res, const Field &x, const Field &y)
    {
        const storage MODULAR = get_modular();

        switch (RADIX_TYPE)
        {
            case RADIX_TYPE_64BIT:
                FIELD_RADIX_64BIT::add_mod_limbs_host<limb_t, storage>(res.limbs_storage, x.limbs_storage, y.limbs_storage, MODULAR, LC);
                break;
        }
    }

    static constexpr inline void sub_mod_limbs(Field &res, const Field &x, const Field &y)
    {
        const storage MODULAR = get_modular();

        switch (RADIX_TYPE)
        {
            case RADIX_TYPE_64BIT:
                FIELD_RADIX_64BIT::sub_mod_limbs_host<limb_t, storage>(res.limbs_storage, x.limbs_storage, y.limbs_storage, MODULAR, LC);
                break;
        }
    }

    static constexpr inline void mul_mod_limbs(Field &res, const Field &x, const Field &y)
    {
        const storage MODULAR = get_modular();
        const limb_t MODULAR_INV = get_modular_inv();

        switch (RADIX_TYPE)
        {
            case RADIX_TYPE_64BIT:
                FIELD_RADIX_64BIT::mul_mod_limbs_host<limb_t, storage, storage_wide>(res.limbs_storage, x.limbs_storage, y.limbs_storage, MODULAR, MODULAR_INV, LC);
                break;
        }
    }

    static constexpr inline void sqr_mod_limbs(Field &res, const Field &x)
    {
        const storage MODULAR = get_modular();
        const limb_t MODULAR_INV = get_modular_inv();

        switch (RADIX_TYPE)
        {
            case RADIX_TYPE_64BIT:
                FIELD_RADIX_64BIT::sqr_mod_limbs_host<limb_t, storage, storage_wide>(res.limbs_storage, x.limbs_storage, MODULAR, MODULAR_INV, LC);
                break;
        }
    }

    static inline Field neg(const Field &x)
    {
        
        const storage MODULAR = get_modular();
        Field res = {};
        Field modular = {};
        modular.limbs_storage = MODULAR;

        sub_mod_limbs(res, modular, x);

        return res;
    }

    static inline bool is_zero(const Field &x)
    {
        for (int i = 0; i < LC; i++)
        {
            if (x.limbs_storage.limbs[i] != 0x0)
            {
                return false;
            }
        }

        return true;
    }

    static constexpr inline bool is_one(const Field &x)
    {
        return x == get_field_one();
    }

    static constexpr inline bool is_unity_one(const Field &x)
    {
        for (unsigned i = 0; i < LC; i++)
        {
            if (i == 0)
            {
                if (x.limbs_storage.limbs[0] == 0x01)
                {
                    continue;
                }
                else
                {
                    return false;
                }
            }

            if (x.limbs_storage.limbs[i] != 0x0)
            {
                return false;
            }
        }

        return true;
    }    

    friend inline Field operator+(Field x, const Field &y)
    {
        Field res = {};
        add_mod_limbs(res, x, y);
        return res;
    }

    friend inline Field operator-(Field x, const Field &y)
    {
        Field res = {};
        sub_mod_limbs(res, x, y);

        return res;
    }

    friend inline Field operator*(const Field &x, const Field &y)
    {
        Field res = {};
        mul_mod_limbs(res, x, y);

        return res;
    }

    friend inline Field operator^(const Field &x, int exponent)
    {
        Field res = {};
        if (exponent == 2)
        {
            // Perform the square operation
            sqr_mod_limbs(res, x);
        }
        else
        {
            // Handle other cases (optional)
            // For example, you could raise to a different exponent or return an error.
        }

        return res;
    }

    friend inline bool operator==(const Field &x, const Field &y)
    {
        for (int i = 0; i < LC; i++)
        {
            if (x.limbs_storage.limbs[i] != y.limbs_storage.limbs[i])
            {
                return false;
            }
        }

        return true;
    }

    friend inline bool operator!=(const Field &x, const Field &y)
    {
        for (int i = 0; i < LC; i++)
        {
            if (x.limbs_storage.limbs[i] != y.limbs_storage.limbs[i])
            {
                return true;
            }
        }

        return false;
    }

    friend inline bool operator>(const Field &left, const Field &right)
    {
        for (int i = LC - 1; i >= 0; --i)
        {
            if (left.limbs_storage.limbs[i] < right.limbs_storage.limbs[i])
            {
                return false;
            }
            else if (left.limbs_storage.limbs[i] > right.limbs_storage.limbs[i])
            {
                return true;
            }
        }

        return false;
    }

    // from montgomery to bigint
    static constexpr inline Field to_bigint(const Field &input)
    {
        return from_montgomery(input);
    }

    // from bigint to montgomery
    static constexpr inline Field from_bigint(const Field &input)
    {
        return to_montgomery(input);
    }

    static constexpr inline Field read_le(const Field &input)
    {
        return from_bigint(input);
    }

    static constexpr inline Field write_le(const Field &input)
    {
        return to_bigint(input);
    }

    static Field inverse(const Field &x)
    {
        Field u = x;
        Field v = get_field_modulus();
        Field b = get_field_R2();
        Field c = get_field_zero();

        const storage MODULAR = get_modular();
        const limb_t MODULAR_INV = get_modular_inv();
        constexpr storage UNITY_ONE = get_unity_one();

        Field res = {};

        switch (RADIX_TYPE)
        {
            case RADIX_TYPE_64BIT:
                res.limbs_storage = FIELD_RADIX_64BIT::inverse<limb_t, storage>(u.limbs_storage, v.limbs_storage, b.limbs_storage, c.limbs_storage, MODULAR, MODULAR_INV, UNITY_ONE, LC, LB);
                break;
        }

        return res;
    }

    static constexpr inline int cmp_bigint(const Field &xR, const Field &yR)
    {
        Field x = to_bigint(xR);
        Field y = to_bigint(yR);

        return x > y;
    }

    static constexpr inline void print(const Field &x)
    {
        printf("\t0x");
        for (int i = LC - 1; i >= 0; i--)
        {
            switch (RADIX_TYPE)
            {
                case RADIX_TYPE_64BIT:
                    printf("%016lx", x.limbs_storage.limbs[i]);
                    break;
            }
            if (i != 0)
            {
                printf("_");
            }
        }
        printf("\n");
    }

    static constexpr inline void print(const char *info, const Field &x)
    {
        printf("%s\t0x", info);
        for (int i = LC - 1; i >= 0; i--)
        {
            switch (RADIX_TYPE)
            {
                case RADIX_TYPE_64BIT:
                    printf("%016lx", x.limbs_storage.limbs[i]);
                    break;
            }

            if (i != 0)
            {
                printf("_");
            }
        }
        printf("\n");
    }
};


#endif