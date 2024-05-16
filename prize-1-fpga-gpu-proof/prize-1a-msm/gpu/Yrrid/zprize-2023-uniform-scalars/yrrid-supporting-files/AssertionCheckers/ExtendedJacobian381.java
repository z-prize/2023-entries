import java.math.BigInteger;

// Assertion checker for Extended Jacobian (aka, xyzz) point representation on BLS12381

public class ExtendedJacobian381 {
  static public BigInteger P=new BigInteger("1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab", 16);

  // chain assertion values
  static public double X=1.001;
  static public double Y=4.999;
  static public double ZZ=1.41;
  static public double ZZZ=1.30;
  static public double scale=9.844;

  static public double roundUp(double bound) {
    return Math.ceil(bound*1000.0)/1000.0;
  }

  static double add(double a, double b) {
    double res=a+b;

    if(res>scale) throw new RuntimeException("Overflow on add");
	return roundUp(res);
  }

  static double sub(double a, double b) {
    double res=a+Math.ceil(b);

    System.out.println("Ceiling: " + Math.ceil(b));

    if(res>scale) throw new RuntimeException("Overflow on sub");
	return roundUp(res);
  }

  static double sub(double a, double b, double add) {
    double res=a+add;

    if(b>add) throw new RuntimeException("Result could be negative");
    if(res>scale) throw new RuntimeException("Overflow on sub");
    return roundUp(res);
  }

  static double mul(double a, double b) {
    double res=a*b/scale + 1.0;

    if(res>scale) throw new RuntimeException("Overflow on mul");
	return roundUp(res);
  }

  static double reduce(double a) {
    return 1.001;
  }

  static public void test_add_xy_xy() {
    // input assertions
    double X1=1.0, Y1=1.0, X2=1.0, Y2=1.0;

    // computation
    double P=sub(X2, X1, 1);
    double R=sub(Y2, Y1, 1);
    double PP=mul(P, P);
    double PPP=mul(P, PP);
    double Q=mul(X1, PP);
    double PPPQQ=add(PPP, add(Q, Q));
    double X3=sub(mul(R, R), PPPQQ, 4);
    X3=reduce(X3);
    double QSUB=sub(Q, X3, 2);
    double Y3=sub(mul(R, QSUB), mul(Y1, PPP), 2);
    double ZZ3=PP;
    double ZZZ3=PPP;

    // output assertions
    System.out.println(X3 + " " + (X3<=X));
    System.out.println(Y3 + " " + (Y3<=Y));
    System.out.println(ZZ3 + " " + (ZZ3<=ZZ));
    System.out.println(ZZZ3 + " " + (ZZZ3<=ZZZ));
    if(X3>X || Y3>Y || ZZ3>ZZ || ZZZ3>ZZZ) throw new RuntimeException("Limit check failed\n");
  }

  static public void test_add_xyzz_xy() {
    // input assertions
    double X1=X, Y1=Y, ZZ1=ZZ, ZZZ1=ZZZ;
    double X2=1.0, Y2=1.0;

    // computation
    double U2=mul(X2, ZZ1);
    double S2=mul(Y2, ZZZ1);
    double P=sub(U2, X1, 2);
    double R=sub(S2, Y1, 5);
    double PP=mul(P, P);
    double PPP=mul(P, PP);
    double Q=mul(X1, PP);
    double PPPQQ=add(PPP, add(Q, Q));
    double X3=sub(mul(R, R), PPPQQ, 5);
    X3=reduce(X3);
    double QSUB=sub(Q, X3, 2);
    double YP=mul(Y1, PPP);
    double Y3=sub(mul(R, QSUB), YP, 2);
    double ZZ3=mul(ZZ1, PP);
    double ZZZ3=mul(ZZZ1, PPP);

    // output assertions
    System.out.println(X3 + " " + (X3<=X));
    System.out.println(Y3 + " " + (Y3<=Y));
    System.out.println(ZZ3 + " " + (ZZ3<=ZZ));
    System.out.println(ZZZ3 + " " + (ZZZ3<=ZZZ));
    if(X3>X || Y3>Y || ZZ3>ZZ || ZZZ3>ZZZ) throw new RuntimeException("Limit check failed\n");
  }

  static public void test_add_xyzz_xyzz() {
    // input assertions
    double X1=X, Y1=Y, ZZ1=ZZ, ZZZ1=ZZZ;
    double X2=X, Y2=Y, ZZ2=ZZ, ZZZ2=ZZZ;

    // computation
    double U1=mul(X1, ZZ2);
    double U2=mul(X2, ZZ1);
    double S1=mul(Y1, ZZZ2);
    double S2=mul(Y2, ZZZ1);
    double P=sub(U2, U1, 2);
    double R=sub(S2, S1, 2);
    double PP=mul(P, P);
    double PPP=mul(P, PP);
    double Q=mul(U1, PP);
    double PPPQQ=add(PPP, add(Q, Q));
    double X3=sub(mul(R, R), PPPQQ, 5);
    X3=reduce(X3);
    double QSUB=sub(Q, X3, 2);
    double Y3=sub(mul(R, QSUB), mul(Y1, PPP), 2);
    double ZZ3=mul(mul(ZZ1, ZZ2), PP);
    double ZZZ3=mul(mul(ZZZ1, ZZZ2), PPP);

    // output assertions
    System.out.println(X3 + " " + (X3<=X));
    System.out.println(Y3 + " " + (Y3<=Y));
    System.out.println(ZZ3 + " " + (ZZ3<=ZZ));
    System.out.println(ZZZ3 + " " + (ZZZ3<=ZZZ));
    if(X3>X || Y3>Y || ZZ3>ZZ || ZZZ3>ZZZ) throw new RuntimeException("Limit check failed\n");
  }

  static public void test_dbl_xy() {
    // input assertions
    double X1=1.0, Y1=1.0;

    // computation, assume a=0
    double U=2*Y1;
    double V=mul(U, U);
    double W=mul(U, V);
    double S=mul(X1, V);
    double SQ=mul(X1, X1);
    double M=add(SQ, add(SQ, SQ));
    double TWOS=add(S, S);
    double X3=sub(mul(M, M), TWOS, 3);
    X3=reduce(X3);
    double SSUB=sub(S, X3, 2);
    double Y3=sub(mul(M, SSUB), mul(W, Y1), 2);
    double ZZ3=V;
    double ZZZ3=W;

    // output assertions
    System.out.println(X3 + " " + (X3<=X));
    System.out.println(Y3 + " " + (Y3<=Y));
    System.out.println(ZZ3 + " " + (ZZ3<=ZZ));
    System.out.println(ZZZ3 + " " + (ZZZ3<=ZZZ));
    if(X3>X || Y3>Y || ZZ3>ZZ || ZZZ3>ZZZ) throw new RuntimeException("Limit check failed\n");
  }

  static public void test_dbl_xyzz() {
    // input assertions
    double X1=X, Y1=Y, ZZ1=ZZ, ZZZ1=ZZZ;

    // computation, assume a=0
    double U=2*Y1;
    U=reduce(U);
    double V=mul(U, U);
    double W=mul(U, V);
    double S=mul(X1, V);
    double SQ=mul(X1, X1);
    double M=add(SQ, add(SQ, SQ));
    double TWOS=add(S, S);
    double X3=sub(mul(M, M), TWOS, 3);
    X3=reduce(X3);
    double SSUB=sub(S, X3, 2);
    double Y3=sub(mul(M, SSUB), mul(W, Y1), 2);
    double ZZ3=mul(V, ZZ1);
    double ZZZ3=mul(W, ZZZ1);

    // output assertions
    System.out.println(X3 + " " + (X3<=X));
    System.out.println(Y3 + " " + (Y3<=Y));
    System.out.println(ZZ3 + " " + (ZZ3<=ZZ));
    System.out.println(ZZZ3 + " " + (ZZZ3<=ZZZ));
    if(X3>X || Y3>Y || ZZ3>ZZ || ZZZ3>ZZZ) throw new RuntimeException("Limit check failed\n");
  }


  static public void main(String[] args) {
    BigInteger R=BigInteger.ONE.shiftLeft(384);

    System.out.println(R.multiply(BigInteger.valueOf(1000)).divide(P));

    test_add_xy_xy();
    test_add_xyzz_xy();
    test_add_xyzz_xyzz();
    test_dbl_xy();
    test_dbl_xyzz();
    System.out.println("All tests passed");
  }
}
