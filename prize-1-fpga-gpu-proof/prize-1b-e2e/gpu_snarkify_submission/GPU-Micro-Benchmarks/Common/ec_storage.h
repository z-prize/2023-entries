/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

namespace ec {

template<class Curve, bool montgomery>
class storage_xy {
  public:
  typedef Curve CURVE;
  storage<fp<Curve>,montgomery> x;
  storage<fp<Curve>,montgomery> y;

  operator impl_xy<Curve,false>() const {
    impl_xy<Curve,false> res;

    if(x.is_zero() && y.is_zero())
      res=impl_xy<Curve,false>::infinity();
    else {
      res.x=(storage<fp<Curve>,false>)x;
      res.y=(storage<fp<Curve>,false>)y;
    }
    return res;
  }

  operator impl_xy<Curve,true>() const {
    impl_xy<Curve,true> res;

    if(x.is_zero() && y.is_zero())
      res=impl_xy<Curve,true>::infinity();
    else {
      res.x=(impl<fp<Curve>,true>)x;
      res.y=(impl<fp<Curve>,true>)y;
    }
    return res;
  }
};

template<class Curve, bool montgomery>
class storage_xyzz {
  public:
  typedef Curve CURVE;
  storage<fp<Curve>,montgomery> x;
  storage<fp<Curve>,montgomery> y;
  storage<fp<Curve>,montgomery> zz;
  storage<fp<Curve>,montgomery> zzz;

  operator impl_xyzz<Curve,false>() const {
    impl_xy<Curve,false> res;

    res.x=(impl<fp<Curve>,false>)x;
    res.y=(impl<fp<Curve>,false>)y;
    res.zz=(impl<fp<Curve>,false>)zz;
    res.zzz=(impl<fp<Curve>,false>)zzz;
    return res;
  }

  operator impl_xyzz<Curve,true>() const {
    impl_xy<Curve,true> res;

    res.x=(storage<fp<Curve>,true>)x;
    res.y=(storage<fp<Curve>,true>)y;
    res.zz=(storage<fp<Curve>,true>)zz;
    res.zzz=(storage<fp<Curve>,true>)zzz;
    return res;
  }
};

} // namespace ec
