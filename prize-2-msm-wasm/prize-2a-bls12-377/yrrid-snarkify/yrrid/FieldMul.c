static inline void pairAndConvert(F128* res, I128* x0, I128* x1) {
  I128 iConst=i64x2_splat(0x4338000000000000L);
  F128 fConst=I2D(iConst);
  I128 x00, x01, x02, x03;
  I128 x10, x11, x12, x13;

  x00=x0[0]; x01=x0[1]; x02=x0[2]; x03=x0[3];
  x10=x1[0]; x11=x1[1]; x12=x1[2]; x13=x1[3];

  res[0]=f64x2_sub(I2D(i64x2_add(i64x2_extract_ll(x00, x10), iConst)), fConst);
  res[1]=f64x2_sub(I2D(i64x2_add(i64x2_extract_hh(x00, x10), iConst)), fConst);
  res[2]=f64x2_sub(I2D(i64x2_add(i64x2_extract_ll(x01, x11), iConst)), fConst);
  res[3]=f64x2_sub(I2D(i64x2_add(i64x2_extract_hh(x01, x11), iConst)), fConst);
  res[4]=f64x2_sub(I2D(i64x2_add(i64x2_extract_ll(x02, x12), iConst)), fConst);
  res[5]=f64x2_sub(I2D(i64x2_add(i64x2_extract_hh(x02, x12), iConst)), fConst);
  res[6]=f64x2_sub(I2D(i64x2_add(i64x2_extract_ll(x03, x13), iConst)), fConst);
  res[7]=f64x2_sub(I2D(i64x2_add(i64x2_extract_hh(x03, x13), iConst)), fConst);
}

void fieldMul(Field_t* r0, Field_t* r1, Field_t* a0, Field_t* b0, Field_t* a1, Field_t* b1) {
  F128     term, c2, c3, c4, a[8], b[8], p[8], lh[8];
  I128     sum[17], c0, c1;
  uint64_t q0, q1;

//fieldResolve(a0); fieldResolve(a1); fieldResolve(b0); fieldResolve(b1);

  sum[0]=i64x2_splat(0x7990000000000000L);
  sum[1]=i64x2_splat(0x66B0000000000000L);
  sum[2]=i64x2_splat(0x53D0000000000000L);
  sum[3]=i64x2_splat(0x40F0000000000000L);
  sum[4]=i64x2_splat(0x2E10000000000000L);
  sum[5]=i64x2_splat(0x1B30000000000000L);
  sum[6]=i64x2_splat(0x0850000000000000L);
  sum[7]=i64x2_splat(0xF570000000000000L);
  sum[8]=i64x2_splat(0xEF70000000000000L);
  sum[9]=i64x2_splat(0x0250000000000000L);
  sum[10]=i64x2_splat(0x1530000000000000L);
  sum[11]=i64x2_splat(0x2810000000000000L);
  sum[12]=i64x2_splat(0x3AF0000000000000L);
  sum[13]=i64x2_splat(0x4DD0000000000000L);
  sum[14]=i64x2_splat(0x60B0000000000000L);
  sum[15]=i64x2_splat(0x7390000000000000L);
  sum[16]=i64x2_splat(0x0L);

  pairAndConvert(a, (I128*)a0, (I128*)a1);
  pairAndConvert(b, (I128*)b0, (I128*)b1);

  c0=i64x2_splat(0xFFFFFFFFFFFFL);
  c1=i64x2_splat(0x4330000000000000L);
  c2=I2D(c1);
  c3=I2D(i64x2_splat(0x4638000000000000L));
  c4=I2D(i64x2_splat(0x4638000000000018L));

  p[0]=f64x2_splat(211106232532993.0);
  p[1]=f64x2_splat(52776558167304.0);
  p[2]=f64x2_splat(79165223820612.0);
  p[3]=f64x2_splat(34030673181193.0);
  p[4]=f64x2_splat(239637716341647.0);
  p[5]=f64x2_splat(119439974144546.0);
  p[6]=f64x2_splat(18600534148544.0);
  p[7]=f64x2_splat(1847813609413.0);

  for(int i=0;i<8;i++) {
    term=a[i];
    for(int j=0;j<8;j++)
      lh[j]=f64x2_fma(term, b[j], c3);
    for(int j=0;j<8;j++)
      sum[j+1]=i64x2_add(sum[j+1], D2I(lh[j]));
    for(int j=0;j<8;j++)
      lh[j]=f64x2_sub(c4, lh[j]);
    for(int j=0;j<8;j++)
      lh[j]=f64x2_fma(term, b[j], lh[j]);
    for(int j=0;j<8;j++)
      sum[j]=i64x2_add(sum[j], D2I(lh[j]));

    q0=i64x2_extract_l(sum[0])*0xBFFFFFFFFFFFL;
    q1=i64x2_extract_h(sum[0])*0xBFFFFFFFFFFFL;
    term=f64x2_sub(I2D(i64x2_add(i64x2_and(i64x2_make(q0, q1), c0), c1)), c2);

    for(int j=0;j<8;j++)
      lh[j]=f64x2_fma(term, p[j], c3);
    for(int j=0;j<8;j++)
      sum[j+1]=i64x2_add(sum[j+1], D2I(lh[j]));
    for(int j=0;j<8;j++)
      lh[j]=f64x2_sub(c4, lh[j]);
    for(int j=0;j<8;j++)
      lh[j]=f64x2_fma(term, p[j], lh[j]);

    sum[0]=i64x2_add(sum[0], D2I(lh[0]));
    sum[1]=i64x2_add(sum[1], D2I(lh[1]));
    sum[0]=i64x2_add(sum[1], i64x2_shr(sum[0], 48));
    for(int j=1;j<7;j++)
      sum[j]=i64x2_add(sum[j+1], D2I(lh[j+1]));
    sum[7]=sum[8];
    sum[8]=sum[i+9];
  }

  for(int i=0;i<7;i++)
    sum[i+8]=i64x2_shr(sum[i], 48);
  for(int i=0;i<7;i++)
    sum[i]=i64x2_and(sum[i], c0);
  for(int i=0;i<7;i++)
    sum[i+1]=i64x2_add(sum[i+1], sum[i+8]);

  ((I128*)r0)[0]=i64x2_extract_ll(sum[0], sum[1]);
  ((I128*)r1)[0]=i64x2_extract_hh(sum[0], sum[1]);
  ((I128*)r0)[1]=i64x2_extract_ll(sum[2], sum[3]);
  ((I128*)r1)[1]=i64x2_extract_hh(sum[2], sum[3]);
  ((I128*)r0)[2]=i64x2_extract_ll(sum[4], sum[5]);
  ((I128*)r1)[2]=i64x2_extract_hh(sum[4], sum[5]);
  ((I128*)r0)[3]=i64x2_extract_ll(sum[6], sum[7]);
  ((I128*)r1)[3]=i64x2_extract_hh(sum[6], sum[7]);
}
