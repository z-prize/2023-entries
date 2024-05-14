/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

typedef enum {
  stageBatchCreator=0,
  stageDelay=1,
  stageDMultiplier=2,
  stageDTreeMultiplier=3,
  stageTreeCopier=4,
  stageInverter=5,
  stageDUntreeMultiplier=6,
  stageEC=7,
  stageCount=8,
} StageType;

typedef enum {
  postingTypeRowUpdate,
  postingTypeWorkQueued,
  postingTypeUnlock,
} PostingType;

typedef enum {
  updateTypeBatchCreator,
  updateTypeReadBuckets,
  updateTypeDMultiplierInverses,
  updateTypeDMultiplierTree,
  updateTypeTree,
  updateTypeTreeCopy,
  updateTypeUntree,
  updateTypeWriteBuckets,
} UpdateType;

typedef enum {
  operationAdd=0,
  operationSubtract=1
} AddSubtractEnum;

class UpdateEntry {
  public:
  uint8_t             windowIndex;
  uint16_t            bucketIndex;
  uint32_t            pointIndex;
  AddSubtractEnum     addSubtract;
  PointXY<CURVE,true> point;
  PointXY<CURVE,true> bucket;
  CURVE::MainField    inverse;
  CURVE::MainField    treeInverse;

  UpdateEntry() {
    windowIndex=0;
    bucketIndex=INVALID_BUCKET;
    pointIndex=0;
    addSubtract=operationAdd;
    point=PointXY<CURVE,true>();
    bucket=PointXY<CURVE,true>();
    inverse=FF<CURVE::MainField,true>::zero();
    treeInverse=FF<CURVE::MainField,true>::zero();
  }

  static UpdateEntry batchCreator(uint8_t windowIndex, uint16_t bucketIndex, uint32_t pointIndex, AddSubtractEnum addSubtract, const PointXY<CURVE,true>& point) {
    UpdateEntry entry;

    entry.windowIndex=windowIndex;
    entry.bucketIndex=bucketIndex;
    entry.pointIndex=pointIndex;
    entry.addSubtract=addSubtract;
    entry.point=point;
    entry.bucket=PointXY<CURVE,true>();
    return entry;
  }

  static UpdateEntry readBucket(PointXY<CURVE,true> bucket) {
    UpdateEntry entry;

    entry.bucket=bucket;
    return entry;
  }

  static UpdateEntry dMultiplier(const typename CURVE::MainField& inverse) {
    UpdateEntry entry;

    entry.inverse=inverse;
    return entry;
  }

  static UpdateEntry dTreeMultiplier(const typename CURVE::MainField& inverse) {
    UpdateEntry entry;

    entry.inverse=inverse;
    return entry;
  }

  static UpdateEntry treeCopy(const typename CURVE::MainField& inverse) {
    UpdateEntry entry;

    entry.inverse=inverse;
    return entry;
  }

  static UpdateEntry writeBucket(uint8_t windowIndex, uint16_t bucketIndex, const PointXY<CURVE,true>& bucket) {
    UpdateEntry entry;

    entry.windowIndex=windowIndex;
    entry.bucketIndex=bucketIndex;
    entry.bucket=bucket;
    return entry;
  }
};

class UpdateRow {
  public:
  UpdateType       updateType;
  uint32_t         slot;
  uint32_t         tripleIndex;
  uint32_t         offset;
  CURVE::MainField treeProduct;
  UpdateEntry      entries[3];

  static UpdateRow batchCreator(uint32_t slot, uint32_t offset, const UpdateEntry& e1, const UpdateEntry& e2, const UpdateEntry& e3) {
    UpdateRow row;

    row.updateType=updateTypeBatchCreator;
    row.slot=slot;
    row.offset=offset;
    row.entries[0]=e1;
    row.entries[1]=e2;
    row.entries[2]=e3;
    return row;
  }

  static UpdateRow readBuckets(uint32_t slot, uint32_t offset, const UpdateEntry& r1, const UpdateEntry& r2, const UpdateEntry& r3) {
    UpdateRow row;

    row.updateType=updateTypeReadBuckets;
    row.slot=slot;
    row.offset=offset;
    row.entries[0]=r1;
    row.entries[1]=r2;
    row.entries[2]=r3;
    return row;
  }

  static UpdateRow dMultiplierInverses(uint32_t slot, uint32_t offset, const UpdateEntry& u1, const UpdateEntry& u2, const UpdateEntry& u3) {
    UpdateRow row;

    row.updateType=updateTypeDMultiplierInverses;
    row.slot=slot;
    row.offset=offset;
    row.entries[0]=u1;
    row.entries[1]=u2;
    row.entries[2]=u3;
    return row;
  }

  static UpdateRow dMultiplierTree(uint32_t slot, uint32_t offset, const UpdateEntry& u1, const UpdateEntry& u2, const UpdateEntry& u3) {
    UpdateRow row;

    row.updateType=updateTypeDMultiplierTree;
    row.slot=slot;
    row.offset=offset;
    row.entries[0]=u1;
    row.entries[1]=u2;
    row.entries[2]=u3;
    return row;
  }

  static UpdateRow dTree(uint32_t slot, uint32_t tripleIndex, uint32_t offset, const typename CURVE::MainField& treeProduct) {
    UpdateRow row;

    row.updateType=updateTypeTree;
    row.slot=slot;
    row.tripleIndex=tripleIndex;
    row.offset=offset;
    row.treeProduct=treeProduct;
    return row;
  }

  static UpdateRow treeCopy(uint32_t slot, uint32_t offset, const UpdateEntry& u1, const UpdateEntry& u2, const UpdateEntry& u3) {
    UpdateRow row;

    row.updateType=updateTypeTreeCopy;
    row.slot=slot;
    row.offset=offset;
    row.entries[0]=u1;
    row.entries[1]=u2;
    row.entries[2]=u3;
    return row;
  }


  static UpdateRow dUntree(uint32_t slot, uint32_t tripleIndex, uint32_t offset, const typename CURVE::MainField& treeProduct) {
    UpdateRow row;

    row.updateType=updateTypeUntree;
    row.slot=slot;
    row.tripleIndex=tripleIndex;
    row.offset=offset;
    row.treeProduct=treeProduct;
    return row;
  }

  static UpdateRow writeBuckets(const UpdateEntry& w1, const UpdateEntry& w2, UpdateEntry& w3) {
    UpdateRow row;

    row.updateType=updateTypeWriteBuckets;
    row.entries[0]=w1;
    row.entries[1]=w2;
    row.entries[2]=w3;
    return row;
  }
};

class Post {
  public:
  PostingType postingType;
  StageType   stage;
  UpdateRow   rowUpdate;

  static Post* newRowUpdate(const UpdateRow& rowUpdate) {
    Post* post=new Post();

    post->postingType=postingTypeRowUpdate;
    post->rowUpdate=rowUpdate;
    return post;
  }

  static Post* newUnlock(const UpdateRow& unlock) {
    Post* post=new Post();

    post->postingType=postingTypeUnlock;
    post->rowUpdate=unlock;
    return post;
  }

  static Post* newWorkQueued(StageType stage) {
    Post* post=new Post();

    post->postingType=postingTypeWorkQueued;
    post->stage=stage;
    return post;
  }
};

