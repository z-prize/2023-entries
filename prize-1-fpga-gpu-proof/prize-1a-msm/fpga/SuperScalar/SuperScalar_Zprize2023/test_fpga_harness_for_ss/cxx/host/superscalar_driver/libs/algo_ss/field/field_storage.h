#ifndef __FIELD_STORAGE_H__
#define __FIELD_STORAGE_H__

#include <cstdint>
#include "common.h"

using namespace std;
using namespace common;

#define TO_LIMB_T(x)  x

template <typename limb_t, unsigned LIMBS_COUNT> struct alignas(8) field_storage
{
  static constexpr unsigned LC = LIMBS_COUNT;
  limb_t limbs[LIMBS_COUNT];
};

template <typename limb_t, unsigned LIMBS_COUNT> struct alignas(8) field_storage_wide
{
  static_assert(LIMBS_COUNT ^ 1);
  static constexpr unsigned LC = LIMBS_COUNT;
  static constexpr unsigned LC2 = LIMBS_COUNT * 2;
  limb_t limbs[LC2];
};


#endif