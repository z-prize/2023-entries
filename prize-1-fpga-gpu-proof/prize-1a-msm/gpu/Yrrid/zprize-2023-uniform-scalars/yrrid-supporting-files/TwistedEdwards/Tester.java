import java.math.*;
import java.util.*;

public class Tester {
  static public void test1() {
    Random r=new Random();
    SWPointXYZZ swAcc=null;
    TEPointXYTZ teAcc=null;
    SWPointXY   swPoint, swResult, teResult;
    TEPointXYT  tePoint;

    System.out.println("Running test 1:");
    for(int i=0;i<1000;i++) {
      swPoint=new SWPointXY().scale(new BigInteger(256, r)).normalize();
      tePoint=TEScaledCurve.convert(swPoint);

      if(swAcc==null)
        swAcc=new SWPointXYZZ(swPoint);
      else
        swAcc=SWCurve.add(swAcc, swPoint);

      if(teAcc==null)
        teAcc=new TEPointXYTZ(tePoint);
      else
        teAcc=TEScaledCurve.add(teAcc, tePoint);
    }
    System.out.println("Points should match:");
    swResult=swAcc.normalize();
    teResult=TEScaledCurve.convert(teAcc.normalize());
    swResult.dump();
    teResult.dump();
    if(!swResult.x.equals(teResult.x) || !swResult.y.equals(swResult.y))
      throw new RuntimeException("Woa, sum of points does not match!");
    System.out.println();
  }

  static public boolean checkCurveEquation(SWPointXY pt) {
    BigInteger xCubed, ySquared;

    xCubed=Field.cube(pt.x);
    ySquared=Field.sqr(pt.y);
    return ySquared.equals(Field.add(xCubed, BigInteger.ONE));
  }

  static public boolean conversionFails(SWPointXY pt) {
    MPointXY mont=TEScaledCurve.toMontgomery(pt);

    return mont.y.signum()==0 || Field.add(mont.x, BigInteger.ONE).signum()==0;
  }

  static public boolean checkNotOnG1(SWPointXY pt) {
    BigInteger curveOrder=new BigInteger("12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001", 16);

    // zz will be 0 if we're on G1
    return pt.scale(curveOrder).zz.signum()!=0;
  }

  static public void test2() {
    Random      r=new Random();
    SWPointXY   randomPointOnG1=new SWPointXY().scale(new BigInteger(256, r)).normalize();
    SWPointXY[] pts=new SWPointXY[]{
        new SWPointXY(new BigInteger("17b62f01197dc4c6cf996f256d489ac9333ccbfe8c0adeaef1096c9bdf2e3d8e89faab521e38583e9033db52f6523ff", 16),
                      new BigInteger("07d6ff5c7338b952b42317a0f6cc62ed2695adb23e585fc0ac765e8156ff4c12780ac5b0abb46589612e0b472e93b5f", 16)),
        new SWPointXY(new BigInteger("17b62f01197dc4c6cf996f256d489ac9333ccbfe8c0adeaef1096c9bdf2e3d8e89faab521e38583e9033db52f6523ff", 16),
                      new BigInteger("130ca50509185559af8d4465d34830c47b97f17dd0f8d93142bfc47a499533eef8ab0e92544b9a7eef5df4b8d16c4a2", 16)),
        new SWPointXY(new BigInteger("1ae3a4617c510eac63b05c06ca1493b1a22d9f300f5138f1ef3622fba094800170b5d44300000008508c00000000000", 16),
                      new BigInteger("0", 16)),
        new SWPointXY(new BigInteger("9b3af05dd14f6ec619aaf7d34594aabc5ed1347970dec00452217cc900000008508c00000000002", 16),
                      new BigInteger("0", 16)),
        new SWPointXY(new BigInteger("1ae3a4617c510eabc8756ba8f8c524eb8882a75cc9bc8e359064ee822fb5bffd1e94577a00000000000000000000000", 16),
                      new BigInteger("0", 16))
    };

    // Check a random point to make sure we get a false here...
    if(!checkCurveEquation(randomPointOnG1))
      throw new RuntimeException("Something is hosed, checkCurveEquation failed!");
    if(conversionFails(randomPointOnG1))
      throw new RuntimeException("Something is hosed, point is not convertable!");
    if(checkNotOnG1(randomPointOnG1))
      throw new RuntimeException("Something is hosed, curve is definitely on G1!");

    // Here we demonstrate 5 distinct points, s.t.
    // 1.   Each point passes the curve equation check
    // 2.   Each point fails conversion to montgomery
    // 3.   Each point is NOT on the G1 curve
    //
    // There are the only 5 points that can't be converted.  We can conclude: all G1 points can be successfully
    // converted to Twisted Edwards space, and there are no special cases to worry about.

    System.out.println("Running test 2:");
    System.out.println("All values should be true!");
    for(int i=0;i<pts.length;i++) {
      System.out.println(checkCurveEquation(pts[i]) + " " + conversionFails(pts[i]) + " " + checkNotOnG1(pts[i]));
      if(!checkCurveEquation(pts[i]) || !conversionFails(pts[i]) || !checkNotOnG1(pts[i]))
        throw new RuntimeException("Something is hosed, expected properties do not hold");
    }
    System.out.println();
  }

  static public void main(String[] args) {
    test1();
    test2();
    System.out.println("ALL TESTS PASSED SUCCESSFULLY!");

    SWPointXY point=new SWPointXY();
    TEScaledCurve.toTwistedEdwards(TEScaledCurve.toMontgomery(point)).dump();
  }
}
