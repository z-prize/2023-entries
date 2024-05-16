import java.math.*;

public class TEScaledCurve {
  // see https://upcommons.upc.edu/bitstream/handle/2117/361741/mathematics-09-03022.pdf, Algorithm 4

  static public BigInteger ONE=BigInteger.ONE;

  static public BigInteger a=Field.P.subtract(ONE);
  static public BigInteger d=new BigInteger("cb5d5818b4d2796505bb385732fe3a1bbc34cc60d19690bf8b2d97f059909cb9aeca3c3871e9f27016092b426b700b", 16);
  static public BigInteger k=d.shiftLeft(1);

  static public BigInteger f=new BigInteger("272fd56ac5c6690cec22e65036018380d743e1f6c15c7cab82b31405cf8a307af39509df5027b6450ae9206343e6e4", 16);
  static public BigInteger fInv=new BigInteger("148a180a7948f239292befe4d8fc837023a5a4061efe45e183a0e17d880d6eb1c87ccf678e91773356175917ba313cb", 16);

  static public BigInteger alpha=Field.P.subtract(ONE);
  static public BigInteger s=new BigInteger("10f272020f118a1dc07a44b1eeea84d7a504665d66cc8c0ff643cca95ccc0d0f793b8504b428d43401d618f0339eab", 16);
  static public BigInteger BInv=new BigInteger("32d756062d349e59416ece15ccbf8e86ef0d33183465a42fe2cb65fc1664272e6bb28f0e1c7a7c9c05824ad09adc01", 16);
  static public BigInteger AOver3B=alpha;

  static public void printParameters() {
    BigInteger te_a=TECurve.a;
    BigInteger te_d=TECurve.d;

    BigInteger scaled_twisted_a=Field.P.subtract(ONE);
    BigInteger scaled_twisted_d=Field.negate(Field.div(te_d, te_a));
    BigInteger scaled_twisted_f=Field.sqrt(Field.negate(te_a));

    System.out.printf("a=%x, i.e., -1\n", scaled_twisted_a);
    System.out.printf("d=%x\n", scaled_twisted_d);
    System.out.printf("f=%x\n", scaled_twisted_f);
    System.out.printf("fInv=%x\n", Field.inv(scaled_twisted_f));
  }

  static public MPointXY toMontgomery(SWPointXY pt) {
    return new MPointXY(Field.mul(s, Field.sub(pt.x, alpha)), Field.mul(s, pt.y));
  }

  static public TEPointXYT toTwistedEdwards(MPointXY pt) {
    return new TEPointXYT(Field.mul(Field.div(pt.x, pt.y), f), Field.div(Field.sub(pt.x, ONE), Field.add(pt.x, ONE)));
  }

  static public MPointXY fromTwistedEdwards(TEPointXYT pt) {
    BigInteger onePlus=Field.add(ONE, pt.y);
    BigInteger oneMinus=Field.sub(ONE, pt.y);

    return new MPointXY(Field.div(onePlus, oneMinus), Field.div(Field.mul(onePlus, f), Field.mul(oneMinus, pt.x)));
  }

  static public SWPointXY fromMontgomery(MPointXY pt) {
    return new SWPointXY(Field.add(Field.mul(pt.x, BInv), AOver3B), Field.mul(pt.y, BInv));
  }

  static public TEPointXYT convert(SWPointXY pt) {
    return toTwistedEdwards(toMontgomery(pt));
  }

  static public SWPointXY convert(TEPointXYT pt) {
    return fromMontgomery(fromTwistedEdwards(pt));
  }

  static public TEPointXYTZ add(TEPointXYTZ p1, TEPointXYT p2) {
    BigInteger A, B, C, D, E, F, G, H, X3, Y3, T3, Z3;

    A=Field.mul(Field.sub(p1.y, p1.x), Field.sub(p2.y, p2.x));
    B=Field.mul(Field.add(p1.y, p1.x), Field.add(p2.y, p2.x));
    C=Field.mul(Field.mul(p1.t, p2.t), k);
    D=Field.add(p1.z, p1.z);
    E=Field.sub(B, A);
    F=Field.sub(D, C);
    G=Field.add(D, C);
    H=Field.add(B, A);
    X3=Field.mul(E, F);
    Y3=Field.mul(G, H);
    T3=Field.mul(E, H);
    Z3=Field.mul(F, G);

    return new TEPointXYTZ(X3, Y3, T3, Z3);
  }

  static public TEPointXYTZ add(TEPointXYTZ p1, TEPointXYTZ p2) {
    BigInteger A, B, C, D, E, F, G, H, X3, Y3, T3, Z3;

    A=Field.mul(Field.sub(p1.y, p1.x), Field.sub(p2.y, p2.x));
    B=Field.mul(Field.add(p1.y, p1.x), Field.add(p2.y, p2.x));
    C=Field.mul(Field.mul(p1.t, p2.t), k);
    D=Field.mul(p1.z, p2.z);
    D=Field.add(D, D);
    E=Field.sub(B, A);
    F=Field.sub(D, C);
    G=Field.add(D, C);
    H=Field.add(B, A);
    X3=Field.mul(E, F);
    Y3=Field.mul(G, H);
    T3=Field.mul(E, H);
    Z3=Field.mul(F, G);

    return new TEPointXYTZ(X3, Y3, T3, Z3);
  }
}
