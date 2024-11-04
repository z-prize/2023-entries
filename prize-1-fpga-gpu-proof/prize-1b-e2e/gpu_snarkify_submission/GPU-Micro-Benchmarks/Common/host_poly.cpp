/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

namespace host {

template<class Field>
class poly_context {
  public:

  static impl<Field,true> evaluate(storage<Field,false>* coefficients, impl<Field,true> x, uint32_t coefficient_count) {
    impl<Field,true> sum=impl<Field,true>::zero();

    for(int32_t i=coefficient_count-1;i>=1;i--)
      sum=(sum + (impl<Field,true>)coefficients[i])*x;
    return sum + (impl<Field,true>)coefficients[0];
  }

  static void divide(storage<Field,false>* r, const storage<Field,false>* p, const uint32_t coefficient_count, const impl<Field,true> x) {
    impl<Field,true> prior=impl<Field,true>::zero(), q;

    // alternate implementation of divide
    // where p[i] are the input coefficients, a is an evaluation point, n is the degree:
    //    s[i] = p[i] * a^i
    //    t[i] = sum{i..n} s[i]     // inclusive suffix sum
    //    r[i] = t[i] * a^-i
    //    q[i] = (r[i]-p[i]) * a^-1

    for(int32_t i=coefficient_count-1;i>=0;i--) {
      q=((impl<Field,true>)p[i]) + prior*x;
      r[i]=(storage<Field,false>)prior;
      prior=q;
    }
  }


  // Witness polynial, w(z_ch):
  //
  //           p(x) - p(z_ch)
  // w(z_ch) = --------------
  //              x - z_ch

  static bool divide_exact(storage<Field,false>* r, const storage<Field,false>* p, const uint32_t coefficient_count, const impl<Field,true> x)  {
    impl<Field,true> inv=x.inv();
    impl<Field,true> q=impl<Field,true>::zero();

    for(uint32_t i=0;i<coefficient_count;i++) {
      q=q-(impl<Field,true>)p[i];
      q=q*inv;
      r[i]=(storage<Field,false>)q;
    }
    if(!q.is_zero())
      fprintf(stderr, "Error polynomial wasn't evenly divisible\n");
    return q.is_zero();
  }
};

} // namespace host
