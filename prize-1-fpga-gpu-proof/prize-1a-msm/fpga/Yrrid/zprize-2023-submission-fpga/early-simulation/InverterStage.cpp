/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

class Inverter {
  public:
  CURVE::MainField A;
  CURVE::MainField B;
  CURVE::MainField U;
  CURVE::MainField V;
  uint64_t         reduceP[16];    // this really should be a static value, oh well
  uint64_t         reduceV[16];
  uint32_t         shiftCount;

  // this is implemented with 6x 64-bit limbs.  In hardware, implemnent this with 16x 24-bit limbs.

  static bool add(typename CURVE::MainField::Words r, const typename CURVE::MainField::Words a, const typename CURVE::MainField::Words b) {
    bool     carry=false;
    uint64_t sum;

    // In HW, use single cycle add trick.  Return value (carry out) comes from the "hierarchy unit" carry out.
    for(int i=0;i<6;i++) {
      sum=a[i] + b[i] + (carry ? 1 : 0);
      carry=(sum<a[i]) || (carry && sum==a[i]);
      r[i]=sum;
    }
    return carry;
  }

  static bool subtract(typename CURVE::MainField::Words r, const typename CURVE::MainField::Words a, const typename CURVE::MainField::Words b) {
    bool     carry=true;
    uint64_t diff;

    // In HW, use single cycle sub trick.  Return value (carry out) comes from the "hierarchy unit" carry out.
    for(int i=0;i<6;i++) {
      diff=a[i] + ~b[i] + (carry ? 1 : 0);
      carry=(diff<a[i]) || (carry && diff==a[i]);
      r[i]=diff;
    }
    return carry;
  }

  static void shiftRight(typename CURVE::MainField::Words x, uint32_t bits) {
    if(bits==0)
      return;
    for(int i=0;i<5;i++)
      x[i]=(x[i]>>bits) | (x[i+1]<<64-bits);
    x[5]=x[5]>>bits;
  }

  static void shiftLeft(typename CURVE::MainField::Words x, uint32_t bits) {
    if(bits==0)
      return;
    for(int i=5;i>=1;i--)
      x[i]=(x[i]<<bits) | (x[i-1]>>64-bits);
    x[0]=x[0]<<bits;
  }

  Inverter() {
    for(int i=0;i<6;i++) {
      A.l[0]=0;
      B.l[0]=0;
      U.l[0]=0;
      V.l[0]=0;
    }
  }

  Inverter(typename CURVE::MainField X) {
    A=X;
    B=CURVE::MainField(CURVE::MainField::prime);
    for(int i=0;i<6;i++) {
      U.l[i]=(i==0) ? 1 : 0;
      V.l[i]=0;
    }
    shiftCount=0;
  }

  bool done() {
    uint64_t lor=A.l[0];

    for(int i=1;i<5;i++)
      lor=lor | A.l[i];
    return lor==0;
  }

  uint8_t shiftAmount(bool zero, bool greaterEqual, bool lessEqual, uint8_t a, uint8_t b) {
    uint8_t bits=0;

    // Note:  b is always odd.
    // Can be a 10-bit lookup table:
    //   address = zero ++ ge ++ le ++ (a & 0x0F) ++ ((b & 0x0F)>>1)
    // possible return values: 0, 1, 2, 3 or 4.

    if(zero || (greaterEqual && lessEqual))
      return 0;

    if((a & 0x01)!=0)
      a=a-b;   // interesting side note, a-b and b-a always have the same number of trailing zeros
    for(int i=0;i<4;i++) {
      if((a & 0x01)==0) {
        bits++;
        a=a>>1;
      }
    }
    return bits;
  }

  bool runStep() {
    CURVE::MainField        aMinusB, bMinusA;
    CURVE::MainField        uMinusV, vMinusU;
    CURVE::MainField::Words ignore, zeroField={0, 0, 0, 0, 0, 0};
    uint32_t                bits;
    bool                    zero, even, greaterEqual, lessEqual;

    // Goal:  runStep works in 2 cycles on the device.

    // use single cycle subtraction for these...
    even=(A.l[0] & 0x01)==0;
    zero=subtract(ignore, zeroField, A.l);             // if 0-A carries out, then A==0
    greaterEqual=subtract(aMinusB.l, A.l, B.l);        // carry out: if A>=B
    lessEqual=subtract(bMinusA.l, B.l, A.l);           // carry out: if A<=B
    subtract(uMinusV.l, U.l, V.l);
    subtract(vMinusU.l, V.l, U.l);

    // this is really a 10-bit table (1024 entries)
    bits=shiftAmount(zero, greaterEqual, lessEqual, A.l[0] & 0x0F, B.l[0] & 0x0F);    // bits will be 0, 1, 2, 3, or 4

    // B, V are two way selections, A and U are 3 way selections
    B=(even || greaterEqual) ? B : A;
    V=(even || greaterEqual) ? V : U;
    A=even ? A : (greaterEqual ? aMinusB : bMinusA);
    U=even ? U : (greaterEqual ? uMinusV : vMinusU);

    // five way selection
    shiftRight(A.l, bits);
    shiftLeft(V.l, bits);

    shiftCount=shiftCount + bits;
    return zero || (lessEqual && greaterEqual);
  }

  void reduceFirst() {
    uint32_t         shift;
    CURVE::MainField prime(CURVE::MainField::prime);

    if(V.l[5]>=0x8000000000000000ull)
      add(V.l, V.l, CURVE::MainField::prime);

    // re-represent prime and V as 16x 24-bit limbs
    for(int i=0;i<16;i++) {
      reduceP[i]=prime.l[0] & 0xFFFFFF;
      shiftRight(prime.l, 24);
      reduceV[i]=V.l[0] & 0xFFFFFF;
      shiftRight(V.l, 24);
    }

    // After the shift step, reduceV[0], reduceV[1], ... reduceV[14] are 24-bits, and reduceV[15] is 40 bits.

    // shift step
    shift=16 - (shiftCount & 0x0F);
    shiftCount=shiftCount + shift;
    reduceV[15]=(reduceV[15]<<shift) | (reduceV[14]>>24-shift);
    for(int i=14;i>=1;i--)
      reduceV[i]=((reduceV[i]<<shift) | (reduceV[i-1]>>24-shift)) & 0xFFFFFF;
    reduceV[0]=(reduceV[0]<<shift) & 0xFFFFFF;
  }

  bool reduceStep() {
    uint32_t l16[16];
    uint32_t m24[16];
    uint32_t h1[16];
    uint64_t q, prod;

    if(shiftCount>0) {
      q=(reduceV[0] * 0xFFFD) & 0xFFFF;

      // reduceV[i] is 25 bits

      // prod can be 41 bits (16 * p[] + 25 bits)
      for(int i=0;i<16;i++) {
        prod=reduceV[i] + reduceP[i]*q;
        l16[i]=prod & 0xFFFF;
        m24[i]=(prod>>16) & 0xFFFFFF;
        h1[i]=(prod>>40);
      }

      // After this step, all reduceV[i] limbs are 25-bits
      reduceV[0]=m24[0] + (l16[1]<<8);
      for(int i=1;i<15;i++)
        reduceV[i]=h1[i-1] + m24[i] + (l16[i+1]<<8);
      reduceV[15]=h1[14] + m24[15] + (h1[15]<<24);

      shiftCount=shiftCount - 16;
    }
    return shiftCount==0;
  }

  CURVE::MainField reduceLast() {
    CURVE::MainField result;

    // resolve any carries
    for(int i=0;i<15;i++) {
      reduceV[i+1]+=reduceV[i]>>24;
      reduceV[i]=reduceV[i] & 0xFFFFFF;
    }

    for(int i=15;i>=0;i--) {
      shiftLeft(result.l, 24);
      result.l[0] += reduceV[i];
    }
    return result;
  }
};

class InverterStage {
  public:
  bool             inputFieldsEmpty;
  bool             outputFieldsEmpty;
  bool             phase1NeedsWork;
  bool             phase2NeedsWork;
  bool             phase2Complete;
  bool             phase3NeedsWork;
  bool             phase3Complete;
  uint32_t         phase1StartCycle;
  uint32_t         phase2StartCycle;
  uint32_t         phase3StartCycle;
  CURVE::MainField input[4];
  Inverter         phase1[4];
  Inverter         phase2[4];
  Inverter         phase3[4];
  CURVE::MainField output[4];

  InverterStage() {
    inputFieldsEmpty=true;
    outputFieldsEmpty=true;
    phase1NeedsWork=true;
    phase2NeedsWork=true;
    phase2Complete=false;
    phase3NeedsWork=true;
    phase3Complete=false;
  }

  bool readyForWork() {
    return inputFieldsEmpty;
  }

  void setInput(uint32_t fieldIndex, typename CURVE::MainField& field) {
    // someone wrote the input fields
    inputFieldsEmpty=false;
    input[fieldIndex]=field;
  }

  typename CURVE::MainField getOutput(uint32_t fieldIndex) {
    // someone read the input fields
    outputFieldsEmpty=true;
    return output[fieldIndex];
  }

  void run(Simulator& simulator, uint32_t cycle) {
    if(outputFieldsEmpty && !phase3NeedsWork && phase3Complete) {
      for(int i=0;i<4;i++)
        output[i]=phase3[i].reduceLast();
      outputFieldsEmpty=false;
      phase3NeedsWork=true;
      simulator.postWork(cycle+2, stageDUntreeMultiplier);
    }
    if(phase3NeedsWork && !phase2NeedsWork && phase2Complete) {
      for(int i=0;i<4;i++) {
        phase3[i]=phase2[i];
        phase3[i].reduceFirst();
      }
      phase3NeedsWork=false;
      phase3Complete=false;
      phase2NeedsWork=true;
      phase3StartCycle=cycle+1;
    }
    if(phase2NeedsWork && !phase1NeedsWork && cycle-phase1StartCycle>=335) {
      for(int i=0;i<4;i++)
        phase2[i]=phase1[i];
      phase2NeedsWork=false;
      phase2Complete=false;
      phase1NeedsWork=true;
      phase2StartCycle=cycle+1;
    }
    if(phase1NeedsWork && !inputFieldsEmpty && simulator.acceptWork(stageInverter)) {
      for(int i=0;i<4;i++)
        phase1[i]=Inverter(input[i]);
      phase1StartCycle=cycle;
      phase1NeedsWork=false;
      inputFieldsEmpty=true;
    }

    if(!phase1NeedsWork && cycle-phase1StartCycle>=8 && cycle-phase1StartCycle<=328) {
      if((cycle-phase1StartCycle)%2==0) {
        for(int i=0;i<4;i++)
          phase1[i].runStep();
      }
    }

    if(!phase2NeedsWork && !phase2Complete && cycle-phase2StartCycle>=8) {
      if((cycle-phase2StartCycle)%2==0) {
        phase2Complete=true;
        for(int i=0;i<4;i++)
          if(!phase2[i].runStep())
            phase2Complete=false;
      }
    }

    if(!phase3NeedsWork && !phase3Complete && cycle-phase3StartCycle>=8) {
      if((cycle-phase3StartCycle)%6==0) {
        phase3Complete=true;
        for(int i=0;i<4;i++)
          if(!phase3[i].reduceStep())
            phase3Complete=false;
      }
    }
  }
};
