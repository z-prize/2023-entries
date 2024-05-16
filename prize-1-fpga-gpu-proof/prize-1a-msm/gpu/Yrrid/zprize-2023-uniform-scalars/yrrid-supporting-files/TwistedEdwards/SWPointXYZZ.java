import java.io.*;
import java.math.BigInteger;

public class SWPointXYZZ {
  public BigInteger x, y, zz, zzz;

  public SWPointXYZZ() {
    this.x=BigInteger.ZERO;
    this.y=BigInteger.ZERO;
    this.zz=BigInteger.ZERO;
    this.zzz=BigInteger.ZERO;
  }

  public SWPointXYZZ(BigInteger x, BigInteger y, BigInteger zz, BigInteger zzz) {
    this.x=x;
    this.y=y;
    this.zz=zz;
    this.zzz=zzz;
  }

  public SWPointXYZZ(SWPointXY copy) {
    this.x=copy.x;
    this.y=copy.y;
    this.zz=BigInteger.ONE;
    this.zzz=BigInteger.ONE;
  }

  public SWPointXYZZ(SWPointXYZZ copy) {
    this.x=copy.x;
    this.y=copy.y;
    this.zz=copy.zz;
    this.zzz=copy.zzz;
  }

  public SWPointXYZZ(BufferedReader reader) throws IOException {
    String line;

    for(int i=0;i<4;i++) {
      line=reader.readLine();
      if(line==null)
        throw new RuntimeException("Unexpected end of stream");
      if(i==0) this.x=new BigInteger(line, 16);
      if(i==1) this.y=new BigInteger(line, 16);
      if(i==2) this.zz=new BigInteger(line, 16);
      if(i==3) this.zzz=new BigInteger(line, 16);
    }
  }

  public void dump() {
    if(zz.equals(BigInteger.ZERO))
      System.out.println("Point at infinity\n");
    else {
      System.out.println("   x = 0x" + x.toString(16));
      System.out.println("   y = 0x" + y.toString(16));
      if(!zz.equals(BigInteger.ONE) || !zzz.equals(BigInteger.ONE)) {
        System.out.println("  zz = 0x" + zz.toString(16));
        System.out.println(" zzz = 0x" + zzz.toString(16));
      }
    }
  }

  public SWPointXYZZ negate() {
    return new SWPointXYZZ(x, Field.negate(y), zz, zzz);
  }

  public SWPointXYZZ add(SWPointXY point) {
    return SWCurve.add(this, point);
  }

  public SWPointXYZZ add(SWPointXYZZ point) {
    return SWCurve.add(this, point);
  }

  public SWPointXYZZ scale(BigInteger amount) {
    SWPointXYZZ acc=new SWPointXYZZ();

    if(amount.signum()==-1)
      throw new RuntimeException("Negative scales not allowed");

    for(int i=amount.bitLength();i>=0;i--) {
      acc=SWCurve.add(acc, acc);
      if(amount.testBit(i))
        acc=SWCurve.add(acc, this);
    }
    return acc;
  }

  public SWPointXYZZ scale(long amount) {
    return scale(BigInteger.valueOf(amount));
  }

  public SWPointXY normalize() {
    if(zz.equals(BigInteger.ZERO))
      throw new RuntimeException("Attempt to normalize point at infinity");
    return new SWPointXY(Field.mul(x, Field.inv(zz)), Field.mul(y, Field.inv(zzz)));
  }
}

