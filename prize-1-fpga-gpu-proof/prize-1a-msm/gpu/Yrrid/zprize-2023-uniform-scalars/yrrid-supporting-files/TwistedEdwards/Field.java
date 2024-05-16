import java.math.*;

public class Field {
  static public int multiplyCount=0;
  static public int squareCount=0;

  static public final BigInteger P=new BigInteger("1ae3a4617c510eac63b05c06ca1493b1a22d9f300f5138f1ef3622fba094800170b5d44300000008508c00000000001", 16);
                                                // 1ae3a4617c510eac63b05c06ca1493b1a22d9f300f5138f1ef3622fba094800170b5d44300000008508c00000000000

  static public void resetCounts() {
    multiplyCount=0;
    squareCount=0;
  }

  static public BigInteger negate(BigInteger x) {
    if(x.signum()==0)
      return BigInteger.ZERO;
    return P.subtract(x);
  }

  static public BigInteger add(BigInteger a, BigInteger b) {
    BigInteger r=a.add(b);

    if(r.compareTo(P)<0)
      return r;
    return r.subtract(P);
  }

  static public BigInteger dbl(BigInteger a) {
    BigInteger r=a.add(a);

    if(r.compareTo(P)<0)
      return r;
    return r.subtract(P);
  }

  static public BigInteger triple(BigInteger a) {
    return add(dbl(a), a);
  }

  static public BigInteger sub(BigInteger a, BigInteger b) {
    BigInteger r=a.subtract(b);

    if(r.signum()>=0)
      return r;
    return r.add(P);
  }

  static public BigInteger mul(BigInteger a, BigInteger b) {
    multiplyCount++;
    return a.multiply(b).remainder(P);
  }

  static public BigInteger sqr(BigInteger a) {
    squareCount++;
    return a.multiply(a).remainder(P);
  }

  static public BigInteger cube(BigInteger a) {
    return mul(sqr(a), a);
  }

  static public BigInteger inv(BigInteger x) {
    return x.modInverse(P);
  }

  static public BigInteger div(BigInteger num, BigInteger denom) {
    return Field.mul(num, Field.inv(denom));
  }

  static public BigInteger sqrt(BigInteger x) {
    // need to implement Tonelli-Shanks!  For now, we cheat, calculating the requested square roots with:
    //    http://www.numbertheory.org/php/tonelli.html
    if(x.equals(BigInteger.valueOf(3)))
      return new BigInteger("30567070899668889872121584789658882274245471728719284894883538395508419196346447682510590835309008936731240225793");
    else if(x.equals(new BigInteger("197530284213631314266409564115575768987902569297476090750117185875703629955647927409947706468955342250977841006594")))
      return new BigInteger("23560188534917577818843641916571445935985386319233886518929971599490231428764380923487987729215299304184915158756");
    throw new RuntimeException("Sqrt '" + x + "' not found in sqrt table");
  }
}

