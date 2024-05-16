import java.io.*;
import java.math.BigInteger;

public class TEPointXYT {
  public BigInteger x, y, t;

  public TEPointXYT() {
    // sometimes you just need some point to work with
    this.x=new BigInteger("767648b9422dde72206f349d4f065b8894aa243d552959b0f23fb9c732ef3fc6585971d2e97493734952a8decc5f31", 16);
    this.y=new BigInteger("a4629e5e925541c842f6662836666f31a521893f198babf955b78a541ff7376d4a72116b634a98e04f356c761573b", 16);
    this.t=Field.mul(this.x, this.y);
  }

  public TEPointXYT(BigInteger x, BigInteger y) {
    this.x=x;
    this.y=y;
    this.t=Field.mul(this.x, this.y);
  }

  public TEPointXYT(BigInteger x, BigInteger y, BigInteger t) {
    this.x=x;
    this.y=y;
    this.t=t;
  }

  public TEPointXYT(TEPointXYT copy) {
    this.x=copy.x;
    this.y=copy.y;
    this.t=copy.t;
  }

  public void dump() {
    System.out.println("   x = 0x" + x.toString(16));
    System.out.println("   y = 0x" + y.toString(16));
    System.out.println("   t = 0x" + t.toString(16));
  }

  public TEPointXYT negate() {
    return new TEPointXYT(Field.negate(x), y, Field.negate(t));
  }

  public TEPointXYTZ add(TEPointXYT point) {
    return TEScaledCurve.add(new TEPointXYTZ(this), point);
  }

  public TEPointXYTZ add(TEPointXYTZ point) {
    return TEScaledCurve.add(point, this);
  }

  public TEPointXYTZ scale(BigInteger amount) {
    TEPointXYTZ acc=new TEPointXYTZ();

    if(amount.signum()==-1)
      throw new RuntimeException("Negative scales not allowed");

    for(int i=amount.bitLength()-1;i>=0;i--) {
      acc=TEScaledCurve.add(acc, acc);
      if(amount.testBit(i))
        acc=TEScaledCurve.add(acc, this);
    }
    return acc;
  }

  public TEPointXYTZ scale(long amount) {
    return scale(BigInteger.valueOf(amount));
  }
}
