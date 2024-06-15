import { FieldMath } from "../../reference/utils/FieldMath";
import { ExtPointType } from "@noble/curves/abstract/edwards";

export const add_points_any_a = (
  a: ExtPointType,
  b: ExtPointType,
  fieldMath: FieldMath,
): ExtPointType => {
  /*
    const A = modP(X1 * X2); // A = X1*X2
    const B = modP(Y1 * Y2); // B = Y1*Y2
    const C = modP(T1 * d * T2); // C = T1*d*T2
    const D = modP(Z1 * Z2); // D = Z1*Z2
    const E = modP((X1 + Y1) * (X2 + Y2) - A - B); // E = (X1+Y1)*(X2+Y2)-A-B
    const F = D - C; // F = D-C
    const G = D + C; // G = D+C
    const H = modP(B - a * A); // H = B-a*A
    const X3 = modP(E * F); // X3 = E*F
    const Y3 = modP(G * H); // Y3 = G*H
    const T3 = modP(E * H); // T3 = E*H
    const Z3 = modP(F * G); // Z3 = F*G
    return new Point(X3, Y3, Z3, T3);
    */
  const ed_a = BigInt(
    "8444461749428370424248824938781546531375899335154063827935233455917409239040",
  );
  const ed_d = BigInt(3021);
  const X1 = a.ex;
  const Y1 = a.ey;
  const T1 = a.et;
  const Z1 = a.ez;
  const X2 = b.ex;
  const Y2 = b.ey;
  const T2 = b.et;
  const Z2 = b.ez;

  const A = fieldMath.Fp.mul(X1, X2);
  const B = fieldMath.Fp.mul(Y1, Y2);
  const C = fieldMath.Fp.mul(fieldMath.Fp.mul(T1, ed_d), T2);
  const D = fieldMath.Fp.mul(Z1, Z2);
  let E = fieldMath.Fp.mul(fieldMath.Fp.add(X1, Y1), fieldMath.Fp.add(X2, Y2));
  E = fieldMath.subtract(E, A);
  E = fieldMath.subtract(E, B);
  const F = fieldMath.Fp.sub(D, C);
  const G = fieldMath.Fp.add(D, C);
  const aA = fieldMath.Fp.mul(ed_a, A);
  const H = fieldMath.Fp.sub(B, aA);
  const X3 = fieldMath.Fp.mul(E, F); // X3 = E*F
  const Y3 = fieldMath.Fp.mul(G, H); // Y3 = G*H
  const T3 = fieldMath.Fp.mul(E, H); // T3 = E*H
  const Z3 = fieldMath.Fp.mul(F, G); // Z3 = F*G
  return fieldMath.createPoint(X3, Y3, T3, Z3);
};

export const add_points_a_minus_one = (
  a: ExtPointType,
  b: ExtPointType,
  fieldMath: FieldMath,
): ExtPointType => {
  const X1 = a.ex;
  const Y1 = a.ey;
  const T1 = a.et;
  const Z1 = a.ez;
  const X2 = b.ex;
  const Y2 = b.ey;
  const T2 = b.et;
  const Z2 = b.ez;

  const A = fieldMath.Fp.mul(
    fieldMath.Fp.sub(Y1, X1),
    fieldMath.Fp.add(Y2, X2),
  );
  const B = fieldMath.Fp.mul(
    fieldMath.Fp.add(Y1, X1),
    fieldMath.Fp.sub(Y2, X2),
  );
  const F = fieldMath.Fp.sub(B, A);

  if (F === BigInt(0)) {
    const A = fieldMath.Fp.mul(X1, X1);
    const B = fieldMath.Fp.mul(Y1, Y1);
    const C = fieldMath.Fp.mul(BigInt(2), fieldMath.Fp.mul(Z1, Z1));
    const D = fieldMath.Fp.mul(BigInt(-1), A);
    const x1y1 = fieldMath.Fp.add(X1, Y1);

    const E = fieldMath.Fp.sub(
      fieldMath.Fp.sub(fieldMath.Fp.mul(x1y1, x1y1), A),
      B,
    );

    const G = fieldMath.Fp.add(D, B);
    const F = fieldMath.Fp.sub(G, C);
    const H = fieldMath.Fp.sub(D, B);

    const X3 = fieldMath.Fp.mul(E, F);
    const Y3 = fieldMath.Fp.mul(G, H);
    const T3 = fieldMath.Fp.mul(E, H);
    const Z3 = fieldMath.Fp.mul(F, G);
    return fieldMath.createPoint(X3, Y3, T3, Z3);
  }

  const C = fieldMath.Fp.mul(fieldMath.Fp.add(Z1, Z1), T2);
  const D = fieldMath.Fp.mul(fieldMath.Fp.add(T1, T1), Z2);
  const E = fieldMath.Fp.add(D, C);
  const G = fieldMath.Fp.add(B, A);
  const H = fieldMath.Fp.sub(D, C);
  const X3 = fieldMath.Fp.mul(E, F);
  const Y3 = fieldMath.Fp.mul(G, H);
  const T3 = fieldMath.Fp.mul(E, H);
  const Z3 = fieldMath.Fp.mul(F, G);
  return fieldMath.createPoint(X3, Y3, T3, Z3);
};
/*
        // Fast algo for doubling Extended Point.
        // https://hyperelliptic.org/EFD/g1p/auto-twisted-extended.html#doubling-dbl-2008-hwcd
        // Cost: 4M + 4S + 1*a + 6add + 1*2.
        double() {
            const { a } = CURVE;
            const { ex: X1, ey: Y1, ez: Z1 } = this;
            const A = modP(X1 * X1); // A = X12
            const B = modP(Y1 * Y1); // B = Y12
            const C = modP(_2n * modP(Z1 * Z1)); // C = 2*Z12
            const D = modP(a * A); // D = a*A
            const x1y1 = X1 + Y1;
            const E = modP(modP(x1y1 * x1y1) - A - B); // E = (X1+Y1)2-A-B
            const G = D + B; // G = D+B
            const F = G - C; // F = G-C
            const H = D - B; // H = D-B
            const X3 = modP(E * F); // X3 = E*F
            const Y3 = modP(G * H); // Y3 = G*H
            const T3 = modP(E * H); // T3 = E*H
            const Z3 = modP(F * G); // Z3 = F*G
            return new Point(X3, Y3, Z3, T3);
        }
*/
