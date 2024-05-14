/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

#include <stdio.h>
#include <stdint.h>

#include "core/core.h"
#include "HexReader.cpp"

typedef FF<BLS12381::OrderField,false> OFF;

int main(int argc, const char** argv) {
  BLS12381::OrderField   power=OFF::zero();
  BLS12381::OrderField   scalar, secret;
  int                    msmCount=1024;
  HexReader              reader("secret.hex");
  const char*            hexBytes;
  uint32_t               amount;
  PointXY<BLS12381,true> result;

  if(argc>1) {
    msmCount=atoi(argv[1]);
  }

  for(int i=0;i<msmCount;i++) {
    if(!reader.skipToHex()) {
      fprintf(stderr, "File 'secret.hex' has too few scalars\n");
      exit(1);
    }
    amount=reader.hexCount(&hexBytes);
    secret=OFF::fromString(hexBytes);
    reader.advance(amount);
    scalar=OFF::random();
    power=OFF::add(power, OFF::mul(scalar, secret));
  }
  result=PointXY<BLS12381,true>::generator();
  result=result.scale(power);
  if(result.infinity)
    printf("Result: Inf\n");
  else {
    printf("X = %016llX%016llX%016llX%016llX%016llX%016llX\n", result.x.l[5], result.x.l[4], result.x.l[3], result.x.l[2], result.x.l[1], result.x.l[0]);
    printf("Y = %016llX%016llX%016llX%016llX%016llX%016llX\n", result.y.l[5], result.y.l[4], result.y.l[3], result.y.l[2], result.y.l[1], result.y.l[0]);
  }
}
