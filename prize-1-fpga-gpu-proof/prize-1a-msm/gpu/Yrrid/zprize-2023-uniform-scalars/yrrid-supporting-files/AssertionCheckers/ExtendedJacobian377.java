// assertion checker for Extended Jacobian (aka, xyzz) for BLS12377

public class ExtendedJacobian377 {
  // chain assertion values
  static public double X=6;
  static public double Y=4;
  static public double ZZ=2;
  static public double ZZZ=2;
  static public double scale=152;

  static public double roundUp(double bound) {
    return Math.ceil(bound*100.0)/100.0;
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
    double QSUB=sub(Q, X3, 6);
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
    double P=sub(U2, X1, 6);
    double R=sub(S2, Y1, 4);
    double PP=mul(P, P);
    double PPP=mul(P, PP);
    double Q=mul(X1, PP);
    double PPPQQ=add(PPP, add(Q, Q));
    double X3=sub(mul(R, R), PPPQQ, 4);
    double QSUB=sub(Q, X3, 6);
    double YP=mul(Y1, PPP);
    double Y3=sub(mul(R, QSUB), YP, 2);
    double ZZ3=mul(ZZ1, PP);
    double ZZZ3=mul(ZZZ1, PPP);

    System.out.println("XYZZ + XY:");
    System.out.println("  U2   = " + U2);
    System.out.println("  S2   = " + S2);
    System.out.println("  P    = " + P);
    System.out.println("  R    = " + R);
    System.out.println("  PP   = " + PP);
    System.out.println("  PPP  = " + PPP);
    System.out.println("  Q    = " + Q);
    System.out.println("  PQ   = " + PPPQQ);
    System.out.println("  X3   = " + X3);
    System.out.println("  QS   = " + QSUB);
    System.out.println("  YP   = " + YP);
    System.out.println("  Y3   = " + Y3);
    System.out.println("  ZZ3  = " + ZZ3);
    System.out.println("  ZZZ3 = " + ZZZ3);


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
    double X3=sub(mul(R, R), PPPQQ, 4);
    double QSUB=sub(Q, X3, 6);
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
    double SSUB=sub(S, X3, 5);
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
    double V=mul(U, U);
    double W=mul(U, V);
    double S=mul(X1, V);
    double SQ=mul(X1, X1);
    double M=add(SQ, add(SQ, SQ));
    double TWOS=add(S, S);
    double X3=sub(mul(M, M), TWOS, 3);
    double SSUB=sub(S, X3, 5);
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
    //test_add_xy_xy();
    test_add_xyzz_xy();
    //test_add_xyzz_xyzz();
    //test_dbl_xy();
    //test_dbl_xyzz();
    System.out.println("All tests passed");
  }
}
