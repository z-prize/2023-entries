import java.io.*;
import java.math.BigInteger;

public class TEPointXYTZ {
  public BigInteger x, y, t, z;

  public TEPointXYTZ() {
    this.x=BigInteger.ZERO;
    this.y=BigInteger.ONE;
    this.t=BigInteger.ZERO;
    this.z=BigInteger.ONE;
  }

  public TEPointXYTZ(BigInteger x, BigInteger y, BigInteger t, BigInteger z) {
    this.x=x;
    this.y=y;
    this.t=t;
    this.z=z;
  }

  public TEPointXYTZ(TEPointXYT copy) {
    this.x=copy.x;
    this.y=copy.y;
    this.t=copy.t;
    this.z=BigInteger.ONE;
  }

  public TEPointXYTZ(TEPointXYTZ copy) {
    this.x=copy.x;
    this.y=copy.y;
    this.t=copy.t;
    this.z=copy.z;
  }

  public void dump() {
    if(z.equals(BigInteger.ZERO))
      System.out.println("Point at infinity\n");
    else {
      System.out.println("   x = 0x" + x.toString(16));
      System.out.println("   y = 0x" + y.toString(16));
      System.out.println("   t = 0x" + t.toString(16));
      if(!z.equals(BigInteger.ONE))
        System.out.println("   z = 0x" + z.toString(16));
    }
  }

  public TEPointXYTZ negate() {
    return new TEPointXYTZ(Field.negate(x), y, Field.negate(t), z);
  }

  public TEPointXYT normalize() {
    BigInteger inv;

    if(z.equals(BigInteger.ZERO))
      throw new RuntimeException("Attempt to normalize point at infinity");
    inv=Field.inv(z);
    return new TEPointXYT(Field.mul(x, inv), Field.mul(y, inv), Field.mul(t, inv));
  }


  public TEPointXYTZ add(TEPointXYT point) {
    return TEScaledCurve.add(this, point);
  }

  public TEPointXYTZ add(TEPointXYTZ point) {
    return TEScaledCurve.add(this, point);
  }

  public TEPointXYTZ scale(BigInteger amount) {
    TEPointXYTZ acc=new TEPointXYTZ();

    if(amount.signum()==-1)
      throw new RuntimeException("Negative scales not allowed");

    for(int i=amount.bitLength();i>=0;i--) {
      acc=TECurve.add(acc, acc);
      if(amount.testBit(i))
        acc=TECurve.add(acc, this);
    }
    return acc;
  }

  public TEPointXYTZ scale(long amount) {
    return scale(BigInteger.valueOf(amount));
  }
}

