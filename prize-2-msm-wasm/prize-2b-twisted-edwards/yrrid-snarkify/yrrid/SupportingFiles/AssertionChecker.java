/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

// This class checks the assertions made in the EC.c.  For a better understanding of this approach,
// please see this talk:  https://www.youtube.com/watch?v=KAWlySN7Hm8, around minute mark, 35:10.

import java.math.BigInteger;

public class AssertionChecker {
  // chain assertion values
  static public double scale=6.85;
  static public double X=1.01;
  static public double Y=1.01;
  static public double T=1.01;
  static public double Z=3.2;

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

  static double reduce(double a) {
    return 1.01;
  }

  static public void test_add_xytz_xyt() {
    double inX=1.0, inY=1.0, inT=1.0;

    double A=mul(sub(Y, X, 2), add(inY, inX));
    double B=mul(add(Y, X), sub(inY, inX, 1));
    double C=mul(Z, inT);
    C=add(C, C);
    double D=add(T, T);
    double E=add(D, C);
    double F=sub(B, A, 2);
    double G=add(B, A);
    double H=sub(D, C, 3);
    double X3=mul(E, F);
    double Y3=mul(G, H);
    double T3=mul(E, H);
    double Z3=mul(F, G);

    X3=reduce(X3);
    Y3=reduce(Y3);
    T3=reduce(T3);

    // output assertions
    System.out.println(X3 + " " + (X3<=X));
    System.out.println(Y3 + " " + (Y3<=Y));
    System.out.println(T3 + " " + (T3<=T));
    System.out.println(Z3 + " " + (Z3<=Z));
    if(X3>X || Y3>Y || T3>T || Z3>Z) throw new RuntimeException("Limit check failed\n");
  }

  static public void test_add_xytz_xytz() {
    double A=mul(sub(Y, X, 2), add(Y, X));
    double B=mul(add(Y, X), sub(Y, X, 2));
    double C=mul(Z, T);
    double D=mul(T, Z);
    C=add(C, C); D=add(D, D);
    double E=add(D, C);
    double F=sub(B, A, 2);
    double G=add(B, A);
    double H=sub(D, C, 3);
    double X3=mul(E, F);
    double Y3=mul(G, H);
    double T3=mul(E, H);
    double Z3=mul(F, G);

    X3=reduce(X3);
    Y3=reduce(Y3);
    T3=reduce(T3);

    // output assertions
    System.out.println(X3 + " " + (X3<=X));
    System.out.println(Y3 + " " + (Y3<=Y));
    System.out.println(T3 + " " + (T3<=T));
    System.out.println(Z3 + " " + (Z3<=Z));
    if(X3>X || Y3>Y || T3>T || Z3>Z) throw new RuntimeException("Limit check failed\n");
  }

  static public void test_add_xytz_xyt_unified() {
    double inX=1.0, inY=1.0, inT=1.0;

    double A=mul(sub(Y, X, 2), sub(inY, inX, 1));
    double B=mul(add(Y, X), add(inY, inX));
    double C=mul(mul(T, 1), 1);
    double D=Z+Z;

    D=reduce(D);

    double E=sub(B, A, 3);
    double F=sub(D, C, 2);
    double G=add(D, C);
    double H=add(B, A);

    double X3=mul(E, F);
    double Y3=mul(G, H);
    double T3=mul(E, H);
    double Z3=mul(F, G);

    X3=reduce(X3);
    Y3=reduce(Y3);
    T3=reduce(T3);

    // output assertions
    System.out.println(X3 + " " + (X3<=X));
    System.out.println(Y3 + " " + (Y3<=Y));
    System.out.println(T3 + " " + (T3<=T));
    System.out.println(Z3 + " " + (Z3<=Z));
    if(X3>X || Y3>Y || T3>T || Z3>Z) throw new RuntimeException("Limit check failed\n");
  }

  static public void test_add_xytz_xytz_unified() {
    double A=mul(sub(Y, X, 2), sub(Y, X, 2));
    double B=mul(add(Y, X), add(Y, X));
    double C=mul(mul(T, T), 1);
    double D=mul(Z, Z);
    D=add(D, D);

    D=reduce(D);

    double E=sub(B, A, 3);
    double F=sub(D, C, 2);
    double G=add(D, C);
    double H=add(B, A);

    double X3=mul(E, F);
    double Y3=mul(G, H);
    double T3=mul(E, H);
    double Z3=mul(F, G);

    X3=reduce(X3);
    Y3=reduce(Y3);
    T3=reduce(T3);

    // output assertions
    System.out.println(X3 + " " + (X3<=X));
    System.out.println(Y3 + " " + (Y3<=Y));
    System.out.println(T3 + " " + (T3<=T));
    System.out.println(Z3 + " " + (Z3<=Z));
    if(X3>X || Y3>Y || T3>T || Z3>Z) throw new RuntimeException("Limit check failed\n");
  }

  static public void main(String[] args) {
    BigInteger P=new BigInteger("12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001", 16);
    double     computedScale=BigInteger.ONE.shiftLeft(51*5+10).divide(P).intValue()/1024.0;

   // System.out.println(computedScale);
    test_add_xytz_xyt();
    test_add_xytz_xytz();
    test_add_xytz_xyt_unified();
    test_add_xytz_xytz_unified();
  }
}
