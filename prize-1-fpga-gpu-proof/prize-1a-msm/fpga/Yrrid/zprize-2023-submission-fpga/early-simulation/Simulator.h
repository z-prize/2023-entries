/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

class WindowStorage {
  public:
  PointXY<CURVE,true> buckets[8192];
};

typedef struct {
  uint8_t  windowIndex;
  uint16_t bucketIndex;
} LoaderRow;

typedef struct {
  bool             pointInfinity;
  CURVE::MainField pointX;
  bool             bucketInfinity;
  CURVE::MainField bucketX;
} DMultRow;

typedef struct {
  uint8_t          windowIndex;
  uint16_t         bucketIndex;
  uint32_t         pointIndex;
  AddSubtractEnum  addSubtract;
  bool             pointInfinity;
  CURVE::MainField pointX;
  CURVE::MainField pointY;
  bool             bucketInfinity;
  CURVE::MainField bucketX;
  CURVE::MainField bucketY;
  CURVE::MainField inverse;
  CURVE::MainField treeInverse;
} ECEntry;

typedef ECEntry ECRow[3];

class BatchCreatorStage;
class DelayStage;
class DMultiplierStage;
class DTreeStage;
class TreeCopierStage;
class InverterStage;
class DTreeCopierStage;
class DUntreeStage;
class ECStage;

using std::multimap;

class Simulator {
  public:
  uint32_t                     cycle;

  uint32_t                     msmIndex;
  uint32_t                     msmCount;
  CURVE::OrderField*           msmScalars;
  PointXY<CURVE,true>*         msmPoints;

  multimap<int,Post*>          postOffice;

  uint8_t                      workQueued[stageCount];

  WindowStorage                windows[WINDOWS];

  bool                         slotBusy[4096/CYCLES];
  uint32_t                     slotBase[4096/CYCLES];

  bool                         dMultiplierPointInfinity[3][4096];
  CURVE::MainField             dMultiplierPointX[3][4096];
  bool                         dMultiplierBucketInfinity[3][4096];
  CURVE::MainField             dMultiplierBucketX[3][4096];

  CURVE::MainField             dTree[3][4096];
  CURVE::MainField             dUntree[3][4096];

  uint8_t                      ecWindowIndex[3][4096];
  uint16_t                     ecBucketIndex[3][4096];
  uint32_t                     ecPointIndex[3][4096];
  AddSubtractEnum              ecAddSubtract[3][4096];
  PointXY<CURVE,true>          ecPoint[3][4096];
  PointXY<CURVE,true>          ecBucket[3][4096];
  CURVE::MainField             ecInverse[3][4096];
  CURVE::MainField             ecTreeInverse[3][4096];

  BatchCreatorStage*           batchCreatorStage;
  DelayStage*                  delayStage;
  DMultiplierStage*            dMultiplierStage;
  DTreeStage*                  dTreeStage;
  TreeCopierStage*             treeCopyierStage;
  InverterStage*               inverterStage;
  DUntreeStage*                dUntreeStage;
  ECStage*                     ecStage;

  Simulator();

  private:
  typename CURVE::OrderField* generateScalars(uint32_t count);
  PointXY<CURVE,true>* readPoints(uint32_t count);
  void processPost(uint32_t cycle, Post* post);
  void processPosts(uint32_t cycle);

  public:
  PointXY<CURVE,true> reduceBuckets();
  void run(int count);
  bool postOfficeEmpty();
  void postRowUpdate(uint32_t arrivalCycle, const UpdateRow& rowUpdate);
  void postUnlock(uint32_t arrivalCycle, const UpdateRow& rowUpdate);
  void postWork(uint32_t arrivalCycle, StageType stage);

  bool acceptWork(StageType stage);

  uint32_t nextSlot(uint32_t slot);
  bool isSlotEmpty(uint32_t slot);
  void markSlotBusy(uint32_t slot);
  void markSlotEmpty(uint32_t slot);
  bool allSlotsEmpty();
  bool getNextScalarAndPoint(CURVE::OrderField& scalar, PointXY<CURVE,true>& point);
  void readBuckets(uint32_t arrivalCycle, uint32_t slot, uint32_t offset, const UpdateEntry& e1, const UpdateEntry& e2, const UpdateEntry& e3);
};
