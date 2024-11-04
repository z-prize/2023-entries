/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

namespace host {

template<class Curve>
class msm_context {
  public:
  impl<ec::xyzz<Curve>>*  buckets;
  storage<ec::xy<Curve>>* points;

  msm_context() {
    buckets=NULL;
  }

  ~msm_context() {
    if(buckets!=NULL)
      free(buckets);
  }

  void zero_buckets() {
    if(buckets==NULL)
      buckets=(impl<ec::xyzz<Curve>>*)malloc(sizeof(impl<ec::xyzz<Curve>>)*16*65536);
    for(int i=0;i<16*65536;i++)
      buckets[i]=impl<ec::xyzz<Curve>>::infinity();
  }

  void slice_scalar(uint32_t* slices, const storage<fr<Curve>,false> scalar) {
    uint64_t mask=0xFFFFull;

    for(int i=0;i<4;i++) {
      slices[0+i*4]=(uint32_t)(scalar.l[i] & mask);
      slices[1+i*4]=(uint32_t)((scalar.l[i]>>16) & mask);
      slices[2+i*4]=(uint32_t)((scalar.l[i]>>32) & mask);
      slices[3+i*4]=(uint32_t)(scalar.l[i]>>48);
    }
  }

  void accumulate_buckets(const storage<fr<Curve>,false>* scalars, const storage<ec::xy<Curve>,false>* points, uint32_t count) {
    impl<ec::xy<Curve>> point;
    uint32_t            slices[16];
    uint32_t            bucket_index;

    fprintf(stderr, "MSM accumulate\n");
    for(uint32_t i=0;i<count;i++) {
      point=(impl<ec::xy<Curve>>)points[i];
      slice_scalar(slices, scalars[i]);
      for(int32_t j=0;j<16;j++) {
        bucket_index=(j<<16)+slices[j];
        if(bucket_index!=0)
          buckets[bucket_index]=buckets[bucket_index] + point;
      }
    }
  }

  void accumulate_buckets(const storage<fr<Curve>,false>* scalars, const storage<ec::xy<Curve>,true>* points, uint32_t count) {
    impl<ec::xy<Curve>> point;
    uint32_t            slices[16];
    uint32_t            bucket_index;

    fprintf(stderr, "MSM accumulate\n");
    for(uint32_t i=0;i<count;i++) {
      point=(impl<ec::xy<Curve>>)points[i];
      slice_scalar(slices, scalars[i]);
      for(int32_t j=0;j<16;j++) {
        bucket_index=(j<<16)+slices[j];
        if(bucket_index!=0)
          buckets[bucket_index]=buckets[bucket_index] + point;
      }
    }
  }

  impl<ec::xy<Curve>> reduce_buckets() {
    impl<ec::xyzz<Curve>> windows[16];
    impl<ec::xyzz<Curve>> sum, sumOfSums;

    fprintf(stderr, "MSM reduce\n");
    for(int32_t w=0;w<16;w++) {
      sum=impl<ec::xyzz<Curve>>::infinity();
      sumOfSums=impl<ec::xyzz<Curve>>::infinity();
      for(int32_t i=65535;i>=1;i--) {
        sum=sum + buckets[w*65536 + i];
        sumOfSums=sumOfSums + sum;
      }
      windows[w]=sumOfSums;
    }

    sum=impl<ec::xyzz<Curve>>::infinity();
    for(int32_t w=15;w>=0;w--) {
      sum=sum.scale(65536);
      sum=sum + windows[w];
    }
    return sum.normalize();
  }
};

} /* host */