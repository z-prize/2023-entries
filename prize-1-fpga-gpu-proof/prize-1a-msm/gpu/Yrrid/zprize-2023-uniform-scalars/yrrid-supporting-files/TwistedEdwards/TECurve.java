import java.math.BigInteger;

public class TECurve {
  static public BigInteger ONE=BigInteger.ONE;
  static public BigInteger TWO=BigInteger.valueOf(2);
  static public BigInteger THREE=BigInteger.valueOf(3);

  static public BigInteger order=SWCurve.order;

  static public BigInteger a=new BigInteger("65aeac0c5a693cb282dd9c2b997f1d0dde1a663068cb485fc596cbf82cc84e5cd7651e1c38f4f9380b0495a135b7ff", 16);
  static public BigInteger d=new BigInteger("1488b9a0b6aa7ae13b828244107ca1e0c44bf8cd08c4846bf2dcb63c1dc7fb1ba33f82613c70b074cfdbb6a5eca47fc", 16);
  static public BigInteger k=d.shiftLeft(1);

  static public BigInteger alpha=Field.P.subtract(ONE);
  static public BigInteger s=new BigInteger("10f272020f118a1dc07a44b1eeea84d7a504665d66cc8c0ff643cca95ccc0d0f793b8504b428d43401d618f0339eab", 16);
  static public BigInteger BInv=new BigInteger("32d756062d349e59416ece15ccbf8e86ef0d33183465a42fe2cb65fc1664272e6bb28f0e1c7a7c9c05824ad09adc01", 16);
  static public BigInteger AOver3B=alpha;

  static public void printParameters() {
    BigInteger sw_a=SWCurve.a;
    BigInteger sw_b=SWCurve.b;
    BigInteger sw_alpha=SWCurve.alpha;

    BigInteger twisted_s=Field.inv(Field.sqrt(Field.add(Field.mul(THREE, Field.sqr(sw_alpha)), sw_a)));      // s = sqrt(3*alpha^2+a)^-1
    BigInteger montgomeryA=Field.mul(Field.mul(THREE, sw_alpha), twisted_s);                                 // A = 3*alpha*s
    BigInteger montgomeryB=twisted_s;                                                                        // B = s
    BigInteger twisted_a=Field.div(Field.add(montgomeryA, TWO), montgomeryB);                                // a=(A+2)/B
    BigInteger twisted_d=Field.div(Field.sub(montgomeryA, TWO), montgomeryB);                                // b=(A-2)/B

    System.out.printf("a=%x\n", twisted_a);
    System.out.printf("d=%x\n", twisted_d);
    System.out.printf("s=%x\n", twisted_s);
    System.out.printf("BInv=%x\n", Field.inv(montgomeryB));
    System.out.printf("AOver3B=%x\n", Field.div(montgomeryA, Field.triple(montgomeryB)));

  }

  static public MPointXY toMontgomery(SWPointXY pt) {
    return new MPointXY(Field.mul(s, Field.sub(pt.x, alpha)), Field.mul(s, pt.y));
  }

  static public TEPointXYT toTwistedEdwards(MPointXY pt) {
    return new TEPointXYT(Field.div(pt.x, pt.y), Field.div(Field.sub(pt.x, ONE), Field.add(pt.x, ONE)));
  }

  static public MPointXY fromTwistedEdwards(TEPointXYT pt) {
    BigInteger onePlus=Field.add(ONE, pt.y);
    BigInteger oneMinus=Field.sub(ONE, pt.y);

    return new MPointXY(Field.div(onePlus, oneMinus), Field.div(onePlus, Field.mul(oneMinus, pt.x)));
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

    A=Field.mul(p1.x, p2.x);
    B=Field.mul(p1.y, p2.y);
    C=Field.mul(Field.mul(p1.t, p2.t), d);
    D=p1.z;
    E=Field.mul(Field.add(p1.x, p1.y), Field.add(p2.x, p2.y));
    E=Field.sub(E, Field.add(A, B));
    F=Field.sub(D, C);
    G=Field.add(D, C);
    H=Field.sub(B, Field.mul(a, A));
    X3=Field.mul(E, F);
    Y3=Field.mul(G, H);
    T3=Field.mul(E, H);
    Z3=Field.mul(F, G);
    return new TEPointXYTZ(X3, Y3, T3, Z3);
  }

  static public TEPointXYTZ add(TEPointXYTZ p1, TEPointXYTZ p2) {
    BigInteger A, B, C, D, E, F, G, H, X3, Y3, T3, Z3;

    A=Field.mul(p1.x, p2.x);
    B=Field.mul(p1.y, p2.y);
    C=Field.mul(Field.mul(p1.t, p2.t), d);
    D=Field.mul(p1.z, p2.z);
    E=Field.mul(Field.add(p1.x, p1.y), Field.add(p2.x, p2.y));
    E=Field.sub(E, Field.add(A, B));
    F=Field.sub(D, C);
    G=Field.add(D, C);
    H=Field.sub(B, Field.mul(a, A));
    X3=Field.mul(E, F);
    Y3=Field.mul(G, H);
    T3=Field.mul(E, H);
    Z3=Field.mul(F, G);
    return new TEPointXYTZ(X3, Y3, T3, Z3);
  }
}