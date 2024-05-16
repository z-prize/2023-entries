import java.math.BigInteger;

public class SWCurve {
  static public BigInteger order=new BigInteger("12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001", 16);

  // curve parameters
  static public BigInteger a=BigInteger.ZERO;
  static public BigInteger b=BigInteger.ONE;
  static public BigInteger alpha=Field.P.subtract(BigInteger.ONE);     // alpha is a root of z^3 + az + b=0  ==>  z^3 = -1  ==>  z=-1

  static public SWPointXYZZ dbl(SWPointXY pt) {
    BigInteger T0, T1, X3, Y3, ZZ3, ZZZ3;

    X3=pt.x; Y3=pt.y;

    T0=Field.add(Y3, Y3);
    ZZ3=Field.sqr(T0);                     // V is in ZZ3
    ZZZ3=Field.mul(T0, ZZ3);               // W is in ZZZ3

    T0=Field.mul(X3, ZZ3);                 // S is in T0
    X3=Field.sqr(X3);                      // X^2 is in X3
    Y3=Field.mul(Y3, ZZZ3);                // W*pt.y is in Y3

    T1=Field.add(X3, X3);                  // M is in T1
    T1=Field.add(X3, T1);

    X3=Field.sqr(T1);                      // M^2 is in X3
    X3=Field.sub(X3, T0);
    X3=Field.sub(X3, T0);
    T0=Field.sub(T0, X3);

    T1=Field.mul(T1, T0);
    Y3=Field.sub(T1, Y3);

    return new SWPointXYZZ(X3, Y3, ZZ3, ZZZ3);
  }

  static public SWPointXYZZ dbl(SWPointXYZZ pt) {
    BigInteger T0, T1, X3, Y3, ZZ3, ZZZ3;

    X3=pt.x; Y3=pt.y; ZZ3=pt.zz; ZZZ3=pt.zzz;

    T1=Field.add(Y3, Y3);
    T0=Field.sqr(T1);                      // V is in T0
    T1=Field.mul(T0, T1);                  // W is in T1

    ZZ3=Field.mul(ZZ3, T0);
    ZZZ3=Field.mul(ZZZ3, T1);

    T0=Field.mul(X3, T0);                  // S is in T0
    X3=Field.sqr(X3);                      // X^2 is in X3
    Y3=Field.mul(Y3, T1);                  // W*pt.y is in Y3

    T1=Field.add(X3, X3);                  // M is in T1
    T1=Field.add(X3, T1);

    X3=Field.sqr(T1);                      // M^2 is in X3
    X3=Field.sub(X3, T0);
    X3=Field.sub(X3, T0);
    T0=Field.sub(T0, X3);

    T1=Field.mul(T1, T0);
    Y3=Field.sub(T1, Y3);

    return new SWPointXYZZ(X3, Y3, ZZ3, ZZZ3);
  }

  static public SWPointXYZZ add(SWPointXY p1, SWPointXY p2) {
    BigInteger T0, T1, X3, Y3, ZZ3, ZZZ3;

    X3=p1.x; Y3=p1.y;

    T0=Field.sub(p2.x, X3);                // P is in T0
    T1=Field.sub(p2.y, Y3);                // R is in T1

    if(T0.signum()==0 && T1.signum()==0)
      return dbl(p2);

    ZZ3=Field.sqr(T0);                     // PP is in ZZ3
    ZZZ3=Field.mul(T0, ZZ3);               // PPP is in ZZZ3

    T0=Field.mul(X3, ZZ3);                 // Q is in T0
    Y3=Field.mul(Y3, ZZZ3);                // Y1*PPP is in Y3

    X3=Field.sqr(T1);                      // R^2 in X3
    X3=Field.sub(X3, ZZZ3);
    X3=Field.sub(X3, T0);
    X3=Field.sub(X3, T0);                  // R^2 - PPP - 2Q is in T0

    T0=Field.sub(T0, X3);
    T0=Field.mul(T1, T0);
    Y3=Field.sub(T0, Y3);

    return new SWPointXYZZ(X3, Y3, ZZ3, ZZZ3);
  }

  static public SWPointXYZZ add(SWPointXYZZ p1, SWPointXY p2) {
    BigInteger T0, T1, T2, X3, Y3, ZZ3, ZZZ3;

    X3=p1.x; Y3=p1.y; ZZ3=p1.zz; ZZZ3=p1.zzz;

    if(p1.zz.signum()==0)
      return new SWPointXYZZ(p2.x, p2.y, BigInteger.ONE, BigInteger.ONE);

    T0=Field.mul(p2.x, ZZ3);
    T1=Field.mul(p2.y, ZZZ3);
    T0=Field.sub(T0, X3);                  // P is in T0
    T1=Field.sub(T1, Y3);                  // R is in T1

    if(T0.signum()==0 && T1.signum()==0)
      return dbl(p2);

    T2=Field.sqr(T0);                      // PP is in T2
    T0=Field.mul(T0, T2);                  // PPP is in T0

    ZZ3=Field.mul(ZZ3, T2);
    ZZZ3=Field.mul(ZZZ3, T0);
    T2=Field.mul(X3, T2);                  // Q is in T2
    Y3=Field.mul(Y3, T0);                  // Y1*PPP is in Y3

    X3=Field.sqr(T1);                      // R^2 in X3
    T0=Field.add(T0, T2);
    T0=Field.add(T0, T2);
    X3=Field.sub(X3, T0);                  // R^2 - PPP - 2Q is in T0

    T2=Field.sub(T2, X3);
    T2=Field.mul(T1, T2);
    Y3=Field.sub(T2, Y3);

    return new SWPointXYZZ(X3, Y3, ZZ3, ZZZ3);
  }

  static public SWPointXYZZ add(SWPointXYZZ p1, SWPointXYZZ p2) {
    BigInteger T0, T1, T2, X3, Y3, ZZ3, ZZZ3;

    X3=p1.x; Y3=p1.y; ZZ3=p1.zz; ZZZ3=p1.zzz;

    if(ZZ3.signum()==0)
      return new SWPointXYZZ(p2);
    if(p2.zz.signum()==0)
      return new SWPointXYZZ(p1);

    X3=Field.mul(X3, p2.zz);
    Y3=Field.mul(Y3, p2.zzz);
    T0=Field.mul(p2.x, ZZ3);
    T1=Field.mul(p2.y, ZZZ3);
    T0=Field.sub(T0, X3);                  // P is in T0
    T1=Field.sub(T1, Y3);                  // R is in T1

    if(T0.signum()==0 && T1.signum()==0)
      return dbl(p2);

    T2=Field.sqr(T0);                      // PP is in T2
    T0=Field.mul(T0, T2);                  // PPP is in T0

    ZZ3=Field.mul(ZZ3, T2);
    ZZ3=Field.mul(ZZ3, p2.zz);
    ZZZ3=Field.mul(ZZZ3, T0);
    ZZZ3=Field.mul(ZZZ3, p2.zzz);

    T2=Field.mul(X3, T2);                  // Q is in T2
    Y3=Field.mul(Y3, T0);                  // Y1*PPP is in Y3

    X3=Field.sqr(T1);                      // R^2 in X3
    T0=Field.add(T0, T2);
    T0=Field.add(T0, T2);
    X3=Field.sub(X3, T0);                  // R^2 - PPP - 2Q is in T0

    T2=Field.sub(T2, X3);
    T2=Field.mul(T1, T2);
    Y3=Field.sub(T2, Y3);

    return new SWPointXYZZ(X3, Y3, ZZ3, ZZZ3);
  }
}