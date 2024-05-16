import java.io.*;
import java.math.BigInteger;

public class SWPointXY {
  public BigInteger x, y;

  public SWPointXY() {
    // sometimes you just need some point to work with
    this.x=new BigInteger("008848defe740a67c8fc6225bf87ff5485951e2caa9d41bb188282c8bd37cb5cd5481512ffcd394eeab9b16eb21be9ef", 16);
    this.y=new BigInteger("01914a69c5102eff1f674f5d30afeec4bd7fb348ca3e52d96d182ad44fb82305c2fe3d3634a9591afd82de55559c8ea6", 16);
  }

  public SWPointXY(BigInteger x, BigInteger y) {
    this.x=x;
    this.y=y;
  }

  public SWPointXY(SWPointXY copy) {
    this.x=copy.x;
    this.y=copy.y;
  }

  public SWPointXY(BufferedReader reader) throws IOException {
    String line;

    line=reader.readLine();
    if(line==null)
      throw new RuntimeException("Unexpected end of stream");
    this.x=new BigInteger(line, 16);
    line=reader.readLine();
    if(line==null)
      throw new RuntimeException("Unexpected end of stream");
    this.y=new BigInteger(line, 16);
  }

  public void dump() {
    System.out.println("   x = 0x" + x.toString(16));
    System.out.println("   y = 0x" + y.toString(16));
  }

  public SWPointXY negate() {
    return new SWPointXY(x, Field.negate(y));
  }

  public SWPointXYZZ add(SWPointXY point) {
    return SWCurve.add(point, this);
  }

  public SWPointXYZZ add(SWPointXYZZ point) {
    return SWCurve.add(point, this);
  }

  public SWPointXYZZ scale(BigInteger amount) {
    SWPointXYZZ acc=new SWPointXYZZ();

    if(amount.signum()==-1)
      throw new RuntimeException("Negative scales not allowed");

    for(int i=amount.bitLength()-1;i>=0;i--) {
      acc=SWCurve.add(acc, acc);
      if(amount.testBit(i))
        acc=SWCurve.add(acc, this);
    }
    return acc;
  }

  public SWPointXYZZ scale(long amount) {
    return scale(BigInteger.valueOf(amount));
  }
}
