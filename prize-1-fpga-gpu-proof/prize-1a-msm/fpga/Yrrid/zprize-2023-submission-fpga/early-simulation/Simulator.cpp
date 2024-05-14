using std::multimap;

Simulator::Simulator() {
  cycle=0;
  msmIndex=0;
  for(int i=0;i<4096/336;i++) {
    slotBusy[i]=false;
    slotBase[i]=i*336;
  }

  for(int i=0;i<WINDOWS;i++)
    for(int j=0;j<8192;j++)
      windows[i].buckets[j]=PointXY<CURVE,true>();
}

PointXY<CURVE,true> Simulator::reduceBuckets() {
  PointXY<CURVE,true> windowSums[18];
  PointXY<CURVE,true> sum, sumOfSums;

  printf("Reducing buckets...\n");

  // fix window 18
  for(int i=8191;i>=8;i--) {
    windows[18].buckets[i & 0x07]=PointXY<CURVE,true>::add(windows[18].buckets[i & 0x07], windows[18].buckets[i]);
    windows[18].buckets[i]=PointXY<CURVE,true>();
  }

  for(int i=0;i<19;i++) {
    sumOfSums=windows[i].buckets[8191];
    sum=windows[i].buckets[8191];
    for(int j=8190;j>=0;j--) {
      sum=PointXY<CURVE,true>::add(sum, windows[i].buckets[j]);
      sumOfSums=PointXY<CURVE,true>::add(sumOfSums, sum);
    }
    windowSums[i]=sumOfSums;
  }

  sum=PointXY<CURVE,true>();
  for(int i=18;i>=0;i--) {
    for(int j=0;j<14;j++)
      sum=PointXY<CURVE,true>::dbl(sum);
    sum=PointXY<CURVE,true>::add(sum, windowSums[i]);
  }
  return sum;
}

typename CURVE::OrderField* Simulator::generateScalars(uint32_t count) {
  typename CURVE::OrderField* scalars=new typename CURVE::OrderField[count];

  printf("Generating scalars...\n");
  for(int i=0;i<count;i++)
    scalars[i]=FF<CURVE::OrderField,false>::random();
  return scalars;
}

PointXY<CURVE,true>* Simulator::readPoints(uint32_t count) {
  PointXY<CURVE,true>* points=new PointXY<CURVE,true>[count];
  HexReader            reader("points.hex");
  const char*          hexBytes;
  uint32_t             amount;

  printf("Loading points.hex...\n");
  for(int i=0;i<count;i++) {
    if(!reader.skipToHex()) {
      fprintf(stderr, "File 'points.hex' has too few points\n");
      exit(1);
    }
    points[i].infinity=false;
    amount=reader.hexCount(&hexBytes);
    points[i].x=FF<CURVE::MainField,true>::fromString(hexBytes);
    reader.advance(amount);
    if(!reader.skipToHex()) {
      fprintf(stderr, "File 'points.hex' has too few points\n");
      exit(1);
    }
    amount=reader.hexCount(&hexBytes);
    points[i].y=FF<CURVE::MainField,true>::fromString(hexBytes);
    reader.advance(amount);
  }
  return points;
}

void Simulator::postRowUpdate(uint32_t arrivalCycle, const UpdateRow& rowUpdate) {
  postOffice.insert({arrivalCycle, Post::newRowUpdate(rowUpdate)});
}

void Simulator::postUnlock(uint32_t arrivalCycle, const UpdateRow& unlock) {
  postOffice.insert({arrivalCycle, Post::newUnlock(unlock)});
}

void Simulator::postWork(uint32_t arrivalCycle, StageType stage) {
  postOffice.insert({arrivalCycle, Post::newWorkQueued(stage)});
}

void Simulator::processPost(uint32_t cycle, Post* post) {
  if(post->postingType==postingTypeRowUpdate) {
    uint32_t location=post->rowUpdate.slot*CYCLES + post->rowUpdate.offset;

    if(post->rowUpdate.updateType==updateTypeBatchCreator) {
      for(int i=0;i<3;i++) {
        dMultiplierPointInfinity[i][location]=post->rowUpdate.entries[i].point.infinity;
        dMultiplierPointX[i][location]=post->rowUpdate.entries[i].point.x;
        ecWindowIndex[i][location]=post->rowUpdate.entries[i].windowIndex;
        ecBucketIndex[i][location]=post->rowUpdate.entries[i].bucketIndex;
        ecPointIndex[i][location]=post->rowUpdate.entries[i].pointIndex;
        ecAddSubtract[i][location]=post->rowUpdate.entries[i].addSubtract;
        ecPoint[i][location]=post->rowUpdate.entries[i].point;
      }
    }
    else if(post->rowUpdate.updateType==updateTypeReadBuckets) {
      for(int i=0;i<3;i++) {
        dMultiplierBucketInfinity[i][location]=post->rowUpdate.entries[i].bucket.infinity;
        dMultiplierBucketX[i][location]=post->rowUpdate.entries[i].bucket.x;
        ecBucket[i][location]=post->rowUpdate.entries[i].bucket;
      }
    }
    else if(post->rowUpdate.updateType==updateTypeDMultiplierInverses) {
      for(int i=0;i<3;i++)
        ecInverse[i][location]=post->rowUpdate.entries[i].inverse;
    }
    else if(post->rowUpdate.updateType==updateTypeDMultiplierTree) {
      for(int i=0;i<3;i++)
        dTree[i][location]=post->rowUpdate.entries[i].inverse;
    }
    else if(post->rowUpdate.updateType==updateTypeTree) {
      dTree[post->rowUpdate.tripleIndex][location]=post->rowUpdate.treeProduct;
    }
    else if(post->rowUpdate.updateType==updateTypeTreeCopy) {
      for(int i=0;i<3;i++)
        dUntree[i][location]=post->rowUpdate.entries[i].inverse;
    }
    else if(post->rowUpdate.updateType==updateTypeUntree) {
      dUntree[post->rowUpdate.tripleIndex][location]=post->rowUpdate.treeProduct;
    }
    else if(post->rowUpdate.updateType==updateTypeWriteBuckets) {
      for(int i=0;i<3;i++) {
        if(post->rowUpdate.entries[i].bucketIndex!=INVALID_BUCKET)
          windows[post->rowUpdate.entries[i].windowIndex].buckets[post->rowUpdate.entries[i].bucketIndex]=post->rowUpdate.entries[i].bucket;
      }
    }
    else {
      printf("UNHANDLED UPDATE TYPE\n");
    }
  }
  else if(post->postingType==postingTypeWorkQueued) {
    workQueued[post->stage]++;
  }
  else if(post->postingType=postingTypeUnlock) {
    batchCreatorStage->unlock(post->rowUpdate);
  }
}

void Simulator::processPosts(uint32_t cycle) {
  multimap<int,Post*>::iterator current, last;

  current=postOffice.lower_bound(cycle);
  last=postOffice.upper_bound(cycle);
  while(current!=last) {
    processPost(cycle, current->second);
    delete(current->second);
    current++;
  }
  postOffice.erase(cycle);
}

bool Simulator::acceptWork(StageType stage) {
  if(workQueued[stage]==0)
    return false;
  workQueued[stage]--;
  return true;
}

void Simulator::run(int count) {
  batchCreatorStage=new BatchCreatorStage(count);
  delayStage=new DelayStage();
  dMultiplierStage=new DMultiplierStage();
  dTreeStage=new DTreeStage(*this);
  treeCopyierStage=new TreeCopierStage();
  inverterStage=new InverterStage();
  dUntreeStage=new DUntreeStage();
  ecStage=new ECStage();

  msmIndex=0;
  msmCount=count;
  msmScalars=generateScalars(count);
  msmPoints=readPoints(1024*256);

  printf("Running...\n");
  while(true) {
    processPosts(cycle);
    if(batchCreatorStage->run(*this, cycle))
      break;
    delayStage->run(*this, cycle);
    dMultiplierStage->run(*this, cycle);
    dTreeStage->run(*this, *inverterStage, cycle);
    treeCopyierStage->run(*this, cycle);
    inverterStage->run(*this, cycle);
    dUntreeStage->run(*this, *inverterStage, cycle);
    ecStage->run(*this, cycle);
    cycle++;
  }
  printf("Run complete...\n");
}

bool Simulator::postOfficeEmpty() {
  return postOffice.size()==0;
}

uint32_t Simulator::nextSlot(uint32_t slot) {
  if(slot==4096/CYCLES-1)
    return 0;
  return slot+1;
}

bool Simulator::isSlotEmpty(uint32_t slot) {
  return !slotBusy[slot];
}

void Simulator::markSlotBusy(uint32_t slot) {
  slotBusy[slot]=true;
}

void Simulator::markSlotEmpty(uint32_t slot) {
  slotBusy[slot]=false;
}

bool Simulator::allSlotsEmpty() {
  for(int i=0;i<4096/CYCLES;i++) {
    if(slotBusy[i])
      return false;
  }
  return true;
}

bool Simulator::getNextScalarAndPoint(CURVE::OrderField& scalar, PointXY<CURVE,true>& point) {
  if(msmIndex>=msmCount)
    return false;
  scalar=msmScalars[msmIndex];
  point=msmPoints[msmIndex];
  msmIndex++;
  return true;
}

void Simulator::readBuckets(uint32_t arrivalCycle, uint32_t slot, uint32_t offset, const UpdateEntry& e1, const UpdateEntry& e2, const UpdateEntry& e3) {
  UpdateEntry r1, r2, r3;

  if(e1.bucketIndex!=INVALID_BUCKET)
    r1=UpdateEntry::readBucket(windows[e1.windowIndex].buckets[e1.bucketIndex]);
  else
    r1=UpdateEntry::readBucket(PointXY<CURVE,true>());

  if(e2.bucketIndex!=INVALID_BUCKET)
    r2=UpdateEntry::readBucket(windows[e2.windowIndex].buckets[e2.bucketIndex]);
  else
    r2=UpdateEntry::readBucket(PointXY<CURVE,true>());

  if(e3.bucketIndex!=INVALID_BUCKET)
    r3=UpdateEntry::readBucket(windows[e3.windowIndex].buckets[e3.bucketIndex]);
  else
    r3=UpdateEntry::readBucket(PointXY<CURVE,true>());

  postRowUpdate(arrivalCycle, UpdateRow::readBuckets(slot, offset, r1, r2, r3));
}
