/***

Copyright (c) 2024, Snarkify, Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

#pragma once
namespace ec {

template<class Curve>
void show(impl<fp<Curve>,true> f) {
  storage<fp<Curve>,false> v=(storage<fp<Curve>,false>)f;

  printf("%016lX%016lX%016lX%016lX%016lX%016lX\n", v.l[5], v.l[4], v.l[3], v.l[2], v.l[1], v.l[0]);
}

template<class Curve, bool montgomery>
class impl_xy {
  public:
  impl<fp<Curve>,montgomery> x, y;

  static impl_xy infinity() {
    impl_xy res;

    res.x=impl<fp<Curve>>::infinity();
    res.y=impl<fp<Curve>>::zero();
    return res;
  }

  static impl_xy generator() {
    impl_xy res;

    res.x=(impl<fp<Curve>,montgomery>)Curve::const_generator_x;
    res.y=(impl<fp<Curve>,montgomery>)Curve::const_generator_y;
    return res;
  }

  bool is_infinity() const {
    return x.is_infinity();
  }

  static bool ec_is_equal(const impl_xy& pt1, const impl_xy& pt2) {
    return pt1.x==pt2.x && pt1.y==pt2.y;
  }

  static impl_xy ec_neg(const impl_xy& pt) {
    impl_xy res;

    res.x=pt.x;
    res.y=-pt.y;
    return res;
  }

  static impl_xy ec_dbl(const impl_xy& pt) {
    impl<fp<Curve>> lambda;
    impl_xy         res;

    if(pt.is_infinity())
      return pt;
    lambda=pt.x.sqr();
    lambda=lambda + lambda + lambda;
    lambda=lambda / pt.y.dbl();

    res.x=lambda.sqr() - pt.x.dbl();
    res.y=lambda * (pt.x - res.x) - pt.y;
    return res;
  }

  static impl_xy ec_add(const impl_xy& pt1, const impl_xy& pt2) {
    impl<fp<Curve>> dx, dy, lambda, twoX;
    impl_xy         res;

    if(pt1.is_infinity()) return pt2;
    if(pt2.is_infinity()) return pt1;

    dx=pt1.x - pt2.x;
    dy=pt1.y - pt2.y;
    if(dx.is_zero()) {
      if(dy.is_zero())
        return ec_dbl(pt1);
      else
        return infinity();
    }
    lambda=dy / dx;

    res.x=lambda.sqr() - (pt1.x + pt2.x);
    res.y=lambda*(pt2.x - res.x) - pt2.y;
    return res;
  }

  bool operator==(const impl_xy& pt) const {
    return ec_is_equal(*this, pt);
  }

  bool operator!=(const impl_xy& pt) const {
    return !ec_is_equal(*this, pt);
  }

  impl_xy operator-() const {
    return ec_neg(*this);
  }

  impl_xy operator+(const impl_xy& pt) const {
    return ec_add(*this, pt);
  }

  impl_xy operator-(const impl_xy& pt) const {
    return ec_add(*this, -pt);
  }

  operator storage_xy<Curve,false>() const {
    storage_xy<Curve,false> res;

    if(is_infinity()) {
      res.x=storage<fp<Curve>,false>::zero();
      res.y=storage<fp<Curve>,false>::zero();
    }
    else {
      res.x=(storage<fp<Curve>,false>)x;
      res.y=(storage<fp<Curve>,false>)y;
    }
    return res;
  }

  operator storage_xy<Curve,true>() const {
    storage_xy<Curve,true> res;

    if(is_infinity()) {
      res.x=storage<fp<Curve>,true>::zero();
      res.y=storage<fp<Curve>,true>::zero();
    }
    else {
      res.x=(storage<fp<Curve>,true>)x;
      res.y=(storage<fp<Curve>,true>)y;
    }
    return res;
  }
};

template<class Curve, bool montgomery>
class impl_xyzz {
  public:
  impl<fp<Curve>,montgomery> x, y, zz, zzz;

  static impl_xyzz infinity() {
    impl_xyzz res;

    res.x=impl<fp<Curve>,montgomery>::zero();
    res.y=impl<fp<Curve>,montgomery>::zero();
    res.zz=impl<fp<Curve>,montgomery>::zero();
    res.zzz=impl<fp<Curve>,montgomery>::zero();
    return res;
  }

  bool is_infinity() const {
    return zz.is_zero();
  }

  static bool ec_is_equal(const impl_xyzz& pt1, const impl_xyzz& pt2) {
    bool pt1_inf=pt1.is_infinity();
    bool pt2_inf=pt2.is_infinity();

    if(!pt1_inf && !pt2_inf)
      return pt1.x*pt2.zz==pt2.x*pt1.zz && pt1.y*pt2.zzz==pt2.y*pt1.zzz;
    else
      return pt1_inf==pt2_inf;
  }

  static impl_xyzz ec_set(const impl_xy<Curve,montgomery>& pt) {
    impl_xyzz res;

    res.x=pt.x;
    res.y=pt.y;
    res.zz=impl<fp<Curve>,montgomery>::one();
    res.zzz=impl<fp<Curve>,montgomery>::one();
    return res;
  }

  static impl_xyzz ec_neg(const impl_xyzz& pt) {
    impl_xyzz res;

    res.x=pt.x;
    res.y=-pt.y;
    res.zz=pt.zz;
    res.zzz=pt.zzz;
    return res;
  }

  static impl_xyzz ec_dbl(const impl_xy<Curve,montgomery>& pt) {
    impl<fp<Curve>,montgomery> U, V, W, S, M;
    impl_xyzz                  res;

    if(pt.is_infinity())
      return infinity();

    U = pt.y.dbl();
    V = U.sqr();
    W = U * V;
    S = pt.x * V;
    M = pt.x.sqr();
    M = M.dbl() + M;
    res.x = M.sqr() - S.dbl();
    res.y = M * (S - res.x) - W * pt.y;
    res.zz = V;
    res.zzz = W;
    return res;
  }

  static impl_xyzz ec_dbl(const impl_xyzz& pt) {
    impl<fp<Curve>,montgomery> U, V, W, S, M;
    impl_xyzz                  res;

    if(pt.is_infinity())
      return infinity();

    U = pt.y.dbl();
    V = U.sqr();
    W = U * V;
    S = pt.x * V;
    M = pt.x.sqr();
    M = M.dbl() + M;
    res.x = M.sqr() - S.dbl();
    res.y = M * (S - res.x) - W * pt.y;
    res.zz = V * pt.zz;
    res.zzz = W * pt.zzz;
    return res;
  }

  static impl_xyzz ec_add(const impl_xy<Curve,montgomery>& pt1, const impl_xy<Curve,montgomery>& pt2) {
    impl<fp<Curve>,montgomery> P, R, PP, PPP, Q;
    impl_xyzz                  res;

    if(pt1.is_infinity())
      return ec_set(pt2);
    if(pt2.is_infinity())
      return ec_set(pt1);

    P = pt2.x - pt1.x;
    R = pt2.y - pt1.y;

    if(P.is_zero() && R.is_zero())
      return ec_dbl(pt2);

    PP = P.sqr();
    PPP = P * PP;
    Q = pt1.x * PP;
    res.x = R.sqr - (PPP + Q.dbl());
    res.y = R * (Q - res.x) - pt1.y * PPP;
    res.zz = PP;
    res.zzz = PPP;
    return res;
  }

  static impl_xyzz ec_add(const impl_xyzz& pt1, const impl_xy<Curve,montgomery>& pt2) {
    impl<fp<Curve>,montgomery> U2, S2, P, R, PP, PPP, Q;
    impl_xyzz                  res;

    if(pt1.is_infinity())
      return ec_set(pt2);
    if(pt2.is_infinity())
      return pt1;

    U2 = pt2.x * pt1.zz;
    S2 = pt2.y * pt1.zzz;
    P = U2 - pt1.x;
    R = S2 - pt1.y;

    if(P.is_zero() && R.is_zero())
      return ec_dbl(pt2);

    PP = P.sqr();
    PPP = P * PP;
    Q = pt1.x * PP;
    res.x = R.sqr() - (PPP + Q.dbl());
    res.y = R * (Q - res.x) - pt1.y * PPP;
    res.zz = pt1.zz * PP;
    res.zzz = pt1.zzz * PPP;
    return res;
  }

  static impl_xyzz ec_add(const impl_xyzz& pt1, const impl_xyzz& pt2) {
    impl<fp<Curve>,montgomery> U1, U2, S1, S2, P, R, PP, PPP, Q;
    impl_xyzz                  res;

    if(pt1.is_infinity())
      return pt2;
    if(pt2.is_infinity())
      return pt1;

    U1 = pt1.x * pt2.zz;
    U2 = pt2.x * pt1.zz;
    S1 = pt1.y * pt2.zzz;
    S2 = pt2.y * pt1.zzz;
    P = U2 - U1;
    R = S2 - S1;

    if(P.is_zero() && R.is_zero())
      return ec_dbl(pt2);

    PP = P.sqr();
    PPP = P * PP;
    Q = U1 * PP;
    res.x = R.sqr() - (PPP + Q.dbl());
    res.y = R * (Q - res.x) - S1 * PPP;
    res.zz = pt1.zz * pt2.zz * PP;
    res.zzz = pt1.zzz * pt2.zzz * PPP;
    return res;
  }

  static impl_xyzz ec_scale(const impl_xy<Curve,true>& pt, const uint64_t k) {
    impl_xyzz res;
    int32_t   bit=63;

    while(bit>=0 && ((k>>bit) & 0x01)==0)
      bit--;
    if(bit<0)
      return impl_xyzz::infinity();
    res=impl_xyzz::ec_set(pt);
    bit--;
    while(bit>=0) {
      res=ec_dbl(res);
      if(((k>>bit) & 0x01)!=0)
        res=res + pt;
      bit--;
    }
    return res;
  }

  static impl_xyzz ec_scale(const impl_xyzz& pt, const uint64_t k) {
    impl_xyzz res=infinity();
    impl_xyzz current=pt;
    uint64_t  local_k=k;

    while(local_k!=0) {
      if((local_k & 0x01)!=0)
        res=res + current;
      current=ec_dbl(current);
      local_k=local_k>>1;
    }
    return res;
  }

/*
  static impl_xyzz ec_scale(const impl& f, const storage<Field, false>& k) {
    impl     res;
    int32_t  bit=Field::bits-1;

    while(bit>=0 && k.l[(bit+1)>>6]==0)
      bit=bit-64;
    while(bit>=0 && !k.test_bit(bit))
      bit--;
    if(bit<0)
      return impl_xyzz::infinity();
    res=f;
    bit--;
    while(bit>=0) {
      res=res.sqr();
      if(k.test_bit(bit))
        res=res * f;
      bit--;
    }
    return res;
  }
*/

  impl_xy<Curve,montgomery> normalize() {
    impl_xy<Curve,montgomery>  res;
    impl<fp<Curve>,montgomery> iii, ii;

    if(is_infinity())
      return impl_xy<Curve,montgomery>::infinity();

    iii = zzz.inv();
    ii = iii * zz;
    ii = ii.sqr();
    res.x = x * ii;
    res.y = y * iii;
    return res;
  }

  impl_xyzz scale(const uint64_t k) const {
    return ec_scale(*this, k);
  }

  bool operator==(const impl_xyzz& pt) const {
    return mod_is_equal(*this, pt);
  }

  bool operator!=(const impl_xyzz& pt) const {
    return !mod_is_equal(*this, pt);
  }

  impl_xyzz operator-() const {
    return ec_neg(*this);
  }

  impl_xyzz operator+(const impl_xy<Curve,montgomery>& pt) const {
    return ec_add(*this, pt);
  }

  impl_xyzz operator-(const impl_xy<Curve,montgomery>& pt) const {
    return ec_add(*this, -pt);
  }

  impl_xyzz operator+(const impl_xyzz& pt) const {
    return ec_add(*this, pt);
  }

  impl_xyzz operator-(const impl_xyzz& pt) const {
    return ec_add(*this, -pt);
  }

  impl_xyzz operator*(const uint64_t k) const {
    return ec_scale(*this, k);
  }

  impl_xyzz operator*(const storage<fr<Curve>,false>& k) const {
    return ec_scale(*this, k);
  }

  operator storage_xyzz<Curve,false>() const {
    storage_xyzz<Curve,false> res;

    res.x=(storage<fr<Curve>,false>)x;
    res.y=(storage<fr<Curve>,false>)y;
    res.zz=(storage<fr<Curve>,false>)zz;
    res.zzz=(storage<fr<Curve>,false>)zzz;
    return res;
  }

  operator storage_xyzz<Curve,true>() const {
    storage_xyzz<Curve,true> res;

    res.x=(storage<fp<Curve>,true>)x;
    res.y=(storage<fp<Curve>,true>)y;
    res.zz=(storage<fp<Curve>,true>)zz;
    res.zzz=(storage<fp<Curve>,true>)zzz;
    return res;
  }
};

} // namespace ec
