/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Written by Niall Emmart.

***/

#include <stdio.h>
#include <stdint.h>

/* Quick and dirty low performance host side XYZZ implementation */

namespace Host {

    struct BLS12384Param {
        uint32_t N[12];
        uint32_t R[12];
        uint32_t RCubed[12];
        uint32_t NP0;
    };

namespace BLS12384 {

    namespace BLS12377_PARAM {

        uint32_t N[12]={
                0x00000001, 0x8508C000, 0x30000000, 0x170B5D44, 0xBA094800, 0x1EF3622F,
                0x00F5138F, 0x1A22D9F3, 0x6CA1493B, 0xC63B05C0, 0x17C510EA, 0x01AE3A46,
        };

        uint32_t R[12]={
                0xFFFFFF68, 0x02CDFFFF, 0x7FFFFFB1, 0x51409F83, 0x8A7D3FF2, 0x9F7DB3A9,     // 2^384 mod N
                0x6E7C6305, 0x7B4E97B7, 0x803C84E8, 0x4CF495BF, 0xE2FDF49A, 0x008D6661,
        };

        uint32_t RCubed[12]={
                0x8815DE20, 0x581F532F, 0xBE329585, 0xE50F4148, 0x0449F513, 0x2BE8B118,     // 2^(384*3) mod N
                0xC804A20E, 0x6A2A9516, 0x13590CB9, 0x3F725407, 0xC0E7DDA5, 0x01065AB4,
        };

        uint32_t NP0=0xFFFFFFFFu;

    };
    namespace BLS12381_PARAM {

        uint32_t N[12]={
                0xffffaaab, 0xb9feffff, 0xb153ffff, 0x1eabfffe,
                0xf6b0f624, 0x6730d2a0, 0xf38512bf, 0x64774b84,
                0x434bacd7, 0x4b1ba7b6, 0x397fe69a, 0x1a0111ea
        };

        uint32_t R[12]={
                0x0002fffd, 0x76090000, 0xc40c0002, 0xebf4000b,
                0x53c758ba, 0x5f489857, 0x70525745, 0x77ce5853,
                0xa256ec6d, 0x5c071a97, 0xfa80e493, 0x15f65ec3
        };

        uint32_t RCubed[12]={
                0xD94CA1E0, 0xED48AC6B, 0x03A7ADF8, 0x315F831E, 0x615E29DD, 0x9A53352A,     // 2^(384*3) mod N
                0x921E1761, 0x34C04E5E, 0x65724728, 0x2512D435, 0x91755D4D, 0x0AA63460,
        };
        uint32_t NP0=0xfffcfffd;

    };

#if 1
//#if USE_381
#define  BLS_PARAM  BLS12381_PARAM
#else
#define  BLS_PARAM  BLS12377_PARAM
#endif

class G1Montgomery {
  public:
  typedef uint32_t Value[12];

  static void setZero(Value& r) {
    for(int i=0;i<12;i++)
      r[i]=0;
  }

  static void setOne(Value& r) {
    for(int i=0;i<12;i++) 
      r[i]=(i==0) ? 1 : 0;
  }

  static void setR(Value& r) {
    for(int i=0;i<12;i++)
      r[i]=BLS_PARAM::R[i];
  }

  static void set(Value& r, const Value& field) {
    for(int i=0;i<12;i++)
      r[i]=field[i];
  }

  static void load(Value& field, uint32_t* ptr) {
    for(int i=0;i<12;i++)
      field[i]=ptr[i];
  }

  static void store(uint32_t* ptr, Value& field) {
    for(int i=0;i<12;i++)
      ptr[i]=field[i];
  }

  static void exportField(uint64_t* ptr, const Value& field) {
    for(int i=0;i<6;i++)
      ptr[i]=(((uint64_t)field[i*2+1])<<32) | ((uint64_t)field[i*2]);
  }

  static bool isZero(const Value& field) {
    for(int i=0;i<12;i++)
      if(field[i]!=0)
        return false;
    return true;
  }

  static bool isGE(const Value& a, const Value& b) {
    int64_t acc=0;

    for(int i=0;i<12;i++) {
      acc+=((uint64_t)a[i]) - ((uint64_t)b[i]);
      acc=acc>>32;
    }
    return acc>=0;
  }

  static void addN(Value& r, const Value& field) {
    int64_t acc=0;

    for(int i=0;i<12;i++) {
      acc+=((uint64_t)field[i]) + ((uint64_t)BLS_PARAM::N[i]);
      r[i]=(uint32_t)acc;
      acc=acc>>32;
    }
  }

  static bool subN(Value& r, const Value& field) {
    int64_t acc=0;

    for(int i=0;i<12;i++) {
      acc+=((uint64_t)field[i]) - ((uint64_t)BLS_PARAM::N[i]);
      r[i]=(uint32_t)acc;
      acc=acc>>32;
    }
    return acc>=0;
  }

  static void add(Value& r, const Value& a, const Value& b) {
    int64_t acc=0;

    for(int i=0;i<12;i++) {
      acc+=((uint64_t)a[i]) + ((uint64_t)b[i]) - ((uint64_t)BLS_PARAM::N[i]);
      r[i]=(uint32_t)acc;
      acc=acc>>32;
    }
    if(acc>=0) 
      return;
    addN(r, r);
  }

  static void sub(Value& r, const Value& a, const Value& b) {
    int64_t acc=0;

    for(int i=0;i<12;i++) {
      acc+=((uint64_t)a[i]) - ((uint64_t)b[i]);
      r[i]=(uint32_t)acc;
      acc=acc>>32;
    }
    if(acc>=0)
      return;
    addN(r, r);
  }

  static void mul(Value& r, const Value& a, const Value& b) {
    uint64_t acc, high=0, q;
    uint32_t res[12];

    for(int i=0;i<12;i++)
      res[i]=0;
    for(int j=0;j<12;j++) {
      acc=0;
      for(int i=0;i<12;i++) {
        acc+=((uint64_t)a[j])*((uint64_t)b[i]) + ((uint64_t)res[i]);
        res[i]=(uint32_t)acc;
        acc=acc>>32;
      }
      high+=acc;
      q=(uint64_t)(res[0]*BLS_PARAM::NP0);
      acc=q*((uint64_t)BLS_PARAM::N[0]) + ((uint64_t)res[0]);
      acc=acc>>32;
      for(int i=1;i<12;i++) {
        acc+=q*((uint64_t)BLS_PARAM::N[i]) + ((uint64_t)res[i]);
        res[i-1]=(uint32_t)acc;
        acc=acc>>32;
      }
      high+=acc;
      res[11]=(uint32_t)high;
      high=high>>32;
    }
    if(high!=0 || isGE(res, BLS_PARAM::N))
      subN(r, res);
    else
      set(r, res);
  }

  static void shiftRight(Value& r, const Value& field, uint32_t bits) {
    uint32_t words=bits>>5;
    uint32_t left;

    if(words>0) {
      for(int i=0;i<12;i++) {
        if(i+words<12) 
          r[i]=field[i+words];
        else
          r[i]=0;
      }
      bits=bits-words*32;
    }
    else {
      for(int i=0;i<12;i++)
        r[i]=field[i];
    }

    if(bits==0)
      return;
    left=32-bits;
    for(int i=0;i<11;i++) 
      r[i]=(r[i]>>bits) | (r[i+1]<<left);
    r[11]=r[11]>>bits;
  }

  static void swap(Value& a, Value& b) {
    uint32_t swap;

    for(int i=0;i<12;i++) {
      swap=a[i];
      a[i]=b[i];
      b[i]=swap;
    }
  }

  static void reduce(Value& r, const Value& field) {
    set(r, field);
    while(isGE(r, BLS_PARAM::N))
      subN(r, r);
  }

  static void print(const Value& field) {
    for(int i=11;i>=0;i--) 
      printf("%08X", field[i]);
    printf("\n");
  }

  static void inverse(Value& r, const Value& field) {
    Value A, B, X, Y;

    // slow, but very easy to code
    set(A, field);
    set(B, BLS_PARAM::N);
    setOne(X);
    setZero(Y); 
    while(!isZero(A)) {
      if((A[0] & 0x01)!=0) {
        if(!isGE(A, B)) {
          swap(A, B);
          swap(X, Y);
        }
        sub(A, A, B);
        sub(X, X, Y);
      }
      shiftRight(A, A, 1);
      if((X[0] & 0x01)!=0)
        addN(X, X);
      shiftRight(X, X, 1);
    }
    mul(r, Y, BLS_PARAM::RCubed);
  }

  static void dump(const Value& field) {
    Value local;

    setOne(local);
    mul(local, local, field);
    for(int i=11;i>=0;i--)
      printf("%08X", local[i]);
    printf("\n");
  }
};


class G1Montgomery2 {
    public:
        typedef uint32_t Value[12];
        const BLS12384Param* param;

        G1Montgomery2(const BLS12384Param* _param) :param(_param) {}
        void setZero(Value& r) {
            for(int i=0;i<12;i++)
                r[i]=0;
        }

        void setOne(Value& r) {
            for(int i=0;i<12;i++)
                r[i]=(i==0) ? 1 : 0;
        }

        void setR(Value& r) {
            for(int i=0;i<12;i++)
                r[i]=param->R[i];
        }

        void set(Value& r, const Value& field) {
            for(int i=0;i<12;i++)
                r[i]=field[i];
        }

        void load(Value& field, uint32_t* ptr) {
            for(int i=0;i<12;i++)
                field[i]=ptr[i];
        }

        void store(uint32_t* ptr, Value& field) {
            for(int i=0;i<12;i++)
                ptr[i]=field[i];
        }

        static void exportField(uint64_t* ptr, const Value& field) {
            for(int i=0;i<6;i++)
                ptr[i]=(((uint64_t)field[i*2+1])<<32) | ((uint64_t)field[i*2]);
        }

        bool isZero(const Value& field) {
            for(int i=0;i<12;i++)
                if(field[i]!=0)
                    return false;
            return true;
        }

        bool isGE(const Value& a, const Value& b) {
            int64_t acc=0;

            for(int i=0;i<12;i++) {
                acc+=((uint64_t)a[i]) - ((uint64_t)b[i]);
                acc=acc>>32;
            }
            return acc>=0;
        }

        void addN(Value& r, const Value& field) {
            int64_t acc=0;

            for(int i=0;i<12;i++) {
                acc+=((uint64_t)field[i]) + ((uint64_t)param->N[i]);
                r[i]=(uint32_t)acc;
                acc=acc>>32;
            }
        }

        bool subN(Value& r, const Value& field) {
            int64_t acc=0;

            for(int i=0;i<12;i++) {
                acc+=((uint64_t)field[i]) - ((uint64_t)param->N[i]);
                r[i]=(uint32_t)acc;
                acc=acc>>32;
            }
            return acc>=0;
        }

        void add(Value& r, const Value& a, const Value& b) {
            int64_t acc=0;

            for(int i=0;i<12;i++) {
                acc+=((uint64_t)a[i]) + ((uint64_t)b[i]) - ((uint64_t)param->N[i]);
                r[i]=(uint32_t)acc;
                acc=acc>>32;
            }
            if(acc>=0)
                return;
            addN(r, r);
        }

        void sub(Value& r, const Value& a, const Value& b) {
            int64_t acc=0;

            for(int i=0;i<12;i++) {
                acc+=((uint64_t)a[i]) - ((uint64_t)b[i]);
                r[i]=(uint32_t)acc;
                acc=acc>>32;
            }
            if(acc>=0)
                return;
            addN(r, r);
        }

        void mul(Value& r, const Value& a, const Value& b) {
            uint64_t acc, high=0, q;
            uint32_t res[12];

            for(int i=0;i<12;i++)
                res[i]=0;
            for(int j=0;j<12;j++) {
                acc=0;
                for(int i=0;i<12;i++) {
                    acc+=((uint64_t)a[j])*((uint64_t)b[i]) + ((uint64_t)res[i]);
                    res[i]=(uint32_t)acc;
                    acc=acc>>32;
                }
                high+=acc;
                q=(uint64_t)(res[0]*param->NP0);
                acc=q*((uint64_t)param->N[0]) + ((uint64_t)res[0]);
                acc=acc>>32;
                for(int i=1;i<12;i++) {
                    acc+=q*((uint64_t)param->N[i]) + ((uint64_t)res[i]);
                    res[i-1]=(uint32_t)acc;
                    acc=acc>>32;
                }
                high+=acc;
                res[11]=(uint32_t)high;
                high=high>>32;
            }
            if(high!=0 || isGE(res, param->N))
                subN(r, res);
            else
                set(r, res);
        }

        void shiftRight(Value& r, const Value& field, uint32_t bits) {
            uint32_t words=bits>>5;
            uint32_t left;

            if(words>0) {
                for(int i=0;i<12;i++) {
                    if(i+words<12)
                        r[i]=field[i+words];
                    else
                        r[i]=0;
                }
                bits=bits-words*32;
            }
            else {
                for(int i=0;i<12;i++)
                    r[i]=field[i];
            }

            if(bits==0)
                return;
            left=32-bits;
            for(int i=0;i<11;i++)
                r[i]=(r[i]>>bits) | (r[i+1]<<left);
            r[11]=r[11]>>bits;
        }

        void swap(Value& a, Value& b) {
            uint32_t swap;

            for(int i=0;i<12;i++) {
                swap=a[i];
                a[i]=b[i];
                b[i]=swap;
            }
        }

        void reduce(Value& r, const Value& field) {
            set(r, field);
            while(isGE(r, param->N))
                subN(r, r);
        }

        void print(const Value& field) {
            for(int i=11;i>=0;i--)
                printf("%08X", field[i]);
            printf("\n");
        }

        void inverse(Value& r, const Value& field) {
            Value A, B, X, Y;

            // slow, but very easy to code
            set(A, field);
            set(B, param->N);
            setOne(X);
            setZero(Y);
            while(!isZero(A)) {
                if((A[0] & 0x01)!=0) {
                    if(!isGE(A, B)) {
                        swap(A, B);
                        swap(X, Y);
                    }
                    sub(A, A, B);
                    sub(X, X, Y);
                }
                shiftRight(A, A, 1);
                if((X[0] & 0x01)!=0)
                    addN(X, X);
                shiftRight(X, X, 1);
            }
            mul(r, Y, param->RCubed);
        }

    };

} /* namespace BLS12384 */

    static const BLS12384Param param_377 = {
        .N = {
            0x00000001, 0x8508C000, 0x30000000, 0x170B5D44, 0xBA094800, 0x1EF3622F,
                    0x00F5138F, 0x1A22D9F3, 0x6CA1493B, 0xC63B05C0, 0x17C510EA, 0x01AE3A46,
        },

        .R = {
            0xFFFFFF68, 0x02CDFFFF, 0x7FFFFFB1, 0x51409F83, 0x8A7D3FF2, 0x9F7DB3A9,     // 2^384 mod N
                    0x6E7C6305, 0x7B4E97B7, 0x803C84E8, 0x4CF495BF, 0xE2FDF49A, 0x008D6661,
        },

        .RCubed = {
            0x8815DE20, 0x581F532F, 0xBE329585, 0xE50F4148, 0x0449F513, 0x2BE8B118,     // 2^(384*3) mod N
                    0xC804A20E, 0x6A2A9516, 0x13590CB9, 0x3F725407, 0xC0E7DDA5, 0x01065AB4,
        },

        .NP0=0xFFFFFFFFu,
        };
    static const BLS12384Param param_381 = {
            .N ={
                0xffffaaab, 0xb9feffff, 0xb153ffff, 0x1eabfffe,
                        0xf6b0f624, 0x6730d2a0, 0xf38512bf, 0x64774b84,
                        0x434bacd7, 0x4b1ba7b6, 0x397fe69a, 0x1a0111ea
            },

            .R ={
                0x0002fffd, 0x76090000, 0xc40c0002, 0xebf4000b,
                        0x53c758ba, 0x5f489857, 0x70525745, 0x77ce5853,
                        0xa256ec6d, 0x5c071a97, 0xfa80e493, 0x15f65ec3
            },

            .RCubed ={
                0xD94CA1E0, 0xED48AC6B, 0x03A7ADF8, 0x315F831E, 0x615E29DD, 0x9A53352A,     // 2^(384*3) mod N
                        0x921E1761, 0x34C04E5E, 0x65724728, 0x2512D435, 0x91755D4D, 0x0AA63460,
            },
            .NP0=0xfffcfffd,
    };

template<class Field>
class PointXYZZ {
  typedef typename Field::Value FieldValue;
  
  public:
  FieldValue x;
  FieldValue y;
  FieldValue zz;
  FieldValue zzz;
  
  PointXYZZ() {
  }

  void reduce() {
    Field::reduce(x, x);
    Field::reduce(y, y);
    Field::reduce(zz, zz);
    Field::reduce(zzz, zzz);
  }

  PointXYZZ(const FieldValue& xValue, const FieldValue& yValue, const FieldValue& zzValue, const FieldValue& zzzValue) {
    Field::set(x, xValue);
    Field::set(y, yValue);
    Field::set(zz, zzValue);
    Field::set(zzz, zzzValue);
  }
  
  void load(uint32_t* ptr) {
    Field::load(x, ptr);
    Field::load(y, ptr + 12);
    Field::load(zz, ptr + 24);
    Field::load(zzz, ptr + 36);
    reduce();
  }

  void store(uint32_t* ptr) {
    Field::store(ptr, x);
    Field::store(ptr + 12, y);
    Field::store(ptr + 24, zz);
    Field::store(ptr + 36, zzz);
  }

  void normalize() {
    FieldValue I;

    if(Field::isZero(zz)) {
      Field::setZero(x);
      Field::setZero(y);
      Field::setZero(zzz);
      return;
    }
    Field::inverse(I, zzz);
    Field::mul(y, y, I);
    Field::mul(I, I, zz);
    Field::mul(I, I, I);
    Field::mul(x, x, I);
    Field::setR(zz);
    Field::setR(zzz);
  }  

  void dump() {
    printf("   x = ");
    Field::dump(x);
    printf("   y = ");
    Field::dump(y);
    printf("  zz = ");
    Field::dump(zz);
    printf(" zzz = ");
    Field::dump(zzz);
  }
};

template<class Field>
class PointXYZZ2 {
    typedef typename Field::Value FieldValue;

public:
    FieldValue x;
    FieldValue y;
    FieldValue zz;
    FieldValue zzz;

    Field fd;
    PointXYZZ2(const BLS12384Param* param): fd(param) {
    }

    void reduce() {
        fd.reduce(x, x);
        fd.reduce(y, y);
        fd.reduce(zz, zz);
        fd.reduce(zzz, zzz);
    }

    void load(uint32_t* ptr) {
        fd.load(x, ptr);
        fd.load(y, ptr + 12);
        fd.load(zz, ptr + 24);
        fd.load(zzz, ptr + 36);
        reduce();
    }

    void store(uint32_t* ptr) {
        fd.store(ptr, x);
        fd.store(ptr + 12, y);
        fd.store(ptr + 24, zz);
        fd.store(ptr + 36, zzz);
    }

    void normalize() {
        FieldValue I;

        if(fd.isZero(zz)) {
            fd.setZero(x);
            fd.setZero(y);
            fd.setZero(zzz);
            return;
        }
        fd.inverse(I, zzz);
        fd.mul(y, y, I);
        fd.mul(I, I, zz);
        fd.mul(I, I, I);
        fd.mul(x, x, I);
        fd.setR(zz);
        fd.setR(zzz);
    }

};

template<class Field>
class AccumulatorXYZZ {
  typedef typename Field::Value FieldValue;

  public:
  PointXYZZ<Field> xyzz;

  AccumulatorXYZZ() {
    Field::setZero(xyzz.zz);
  }
  
  void set(const FieldValue& x, const FieldValue& y, const FieldValue& zz, const FieldValue& zzz) {
    Field::set(xyzz.x, x);
    Field::set(xyzz.y, y);
    Field::set(xyzz.zz, zz);
    Field::set(xyzz.zzz, zzz);
  }

  void setZero() {
    Field::setZero(xyzz.zz);
  }

  void dbl(const PointXYZZ<Field>& point) {
    FieldValue U, V, W, S, M, T, X, Y, ZZ, ZZZ;

    if(Field::isZero(point.zz)) {
      Field::setZero(xyzz.zz);
      return;
    }

    Field::add(U, point.y, point.y);
    Field::mul(V, U, U);
    Field::mul(W, U, V);
    Field::mul(ZZ, V, point.zz);
    Field::mul(ZZZ, W, point.zzz);
    Field::mul(S, point.x, V);
    Field::mul(M, point.x, point.x);
    Field::add(T, M, M);
    Field::add(M, T, M);
    Field::mul(X, M, M);
    Field::sub(X, X, S);
    Field::sub(X, X, S);
    Field::sub(T, S, X);
    Field::mul(Y, M, T);
    Field::mul(T, W, point.y);
    Field::sub(Y, Y, T);
    set(X, Y, ZZ, ZZZ);
  }

  void add(const PointXYZZ<Field>& point) {
    FieldValue U1, U2, S1, S2, P, R, PP, PPP, Q, T, X, Y, ZZ, ZZZ;

    if(Field::isZero(point.zz))
      return;

    if(Field::isZero(xyzz.zz)) {
      set(point.x, point.y, point.zz, point.zzz);
      return;
    }

    Field::mul(U1, xyzz.x, point.zz);
    Field::mul(U2, point.x, xyzz.zz);
    Field::mul(S1, xyzz.y, point.zzz);
    Field::mul(S2, point.y, xyzz.zzz);
    Field::sub(P, U2, U1);
    Field::sub(R, S2, S1);
    if(Field::isZero(P) && Field::isZero(R)) {
      dbl(point);
      return;
    }
    Field::mul(PP, P, P);
    Field::mul(PPP, PP, P);
    Field::mul(Q, U1, PP);
    Field::mul(X, R, R);
    Field::sub(X, X, PPP);
    Field::sub(X, X, Q);
    Field::sub(X, X, Q);
    Field::sub(T, Q, X);
    Field::mul(Y, R, T);
    Field::mul(T, S1, PPP);
    Field::sub(Y, Y, T);
    Field::mul(ZZ, xyzz.zz, point.zz);
    Field::mul(ZZ, ZZ, PP);
    Field::mul(ZZZ, xyzz.zzz, point.zzz);
    Field::mul(ZZZ, ZZZ, PPP);
    set(X, Y, ZZ, ZZZ);
  }
};


template<class Field>
class AccumulatorXYZZ2 {
    typedef typename Field::Value FieldValue;

public:
    PointXYZZ2<Field> xyzz;
    Field fd;
    AccumulatorXYZZ2(const BLS12384Param* param): fd(param), xyzz(param) {
        fd.setZero(xyzz.zz);
    }

    void set(const FieldValue& x, const FieldValue& y, const FieldValue& zz, const FieldValue& zzz) {
        fd.set(xyzz.x, x);
        fd.set(xyzz.y, y);
        fd.set(xyzz.zz, zz);
        fd.set(xyzz.zzz, zzz);
    }

    void setZero() {
        fd.setZero(xyzz.zz);
    }

    void dbl(const PointXYZZ2<Field>& point) {
        FieldValue U, V, W, S, M, T, X, Y, ZZ, ZZZ;

        if(fd.isZero(point.zz)) {
            fd.setZero(xyzz.zz);
            return;
        }

        fd.add(U, point.y, point.y);
        fd.mul(V, U, U);
        fd.mul(W, U, V);
        fd.mul(ZZ, V, point.zz);
        fd.mul(ZZZ, W, point.zzz);
        fd.mul(S, point.x, V);
        fd.mul(M, point.x, point.x);
        fd.add(T, M, M);
        fd.add(M, T, M);
        fd.mul(X, M, M);
        fd.sub(X, X, S);
        fd.sub(X, X, S);
        fd.sub(T, S, X);
        fd.mul(Y, M, T);
        fd.mul(T, W, point.y);
        fd.sub(Y, Y, T);
        set(X, Y, ZZ, ZZZ);
    }

    void add(const PointXYZZ2<Field>& point) {
        FieldValue U1, U2, S1, S2, P, R, PP, PPP, Q, T, X, Y, ZZ, ZZZ;

        if(fd.isZero(point.zz))
            return;

        if(fd.isZero(xyzz.zz)) {
            set(point.x, point.y, point.zz, point.zzz);
            return;
        }

        fd.mul(U1, xyzz.x, point.zz);
        fd.mul(U2, point.x, xyzz.zz);
        fd.mul(S1, xyzz.y, point.zzz);
        fd.mul(S2, point.y, xyzz.zzz);
        fd.sub(P, U2, U1);
        fd.sub(R, S2, S1);
        if(fd.isZero(P) && fd.isZero(R)) {
            dbl(point);
            return;
        }
        fd.mul(PP, P, P);
        fd.mul(PPP, PP, P);
        fd.mul(Q, U1, PP);
        fd.mul(X, R, R);
        fd.sub(X, X, PPP);
        fd.sub(X, X, Q);
        fd.sub(X, X, Q);
        fd.sub(T, Q, X);
        fd.mul(Y, R, T);
        fd.mul(T, S1, PPP);
        fd.sub(Y, Y, T);
        fd.mul(ZZ, xyzz.zz, point.zz);
        fd.mul(ZZ, ZZ, PP);
        fd.mul(ZZZ, xyzz.zzz, point.zzz);
        fd.mul(ZZZ, ZZZ, PPP);
        set(X, Y, ZZ, ZZZ);
    }
};

} /* namespace Host */
