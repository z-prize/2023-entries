/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

void pointXYT_prepare(I128* r, I128* pt, bool negate);                                        // output XYT (7x I128)
void pointXYTZ_zero(I128* r);                                                                 // output XYTZ (10x I128)
void pointXYTZ_prepare(I128* r, I128* pt, bool negate);                                       // output XYTZ (10x I128)

void pointPairXYT_load(PointPairXYT_t* pair, I128* p1, I128* p2);                             // load XYT : XYT (7x I128) -> PointPairXYT
void pointPairXYT_store(I128* p1, I128* p2, PointPairXYT_t* pair);                            // store PointPairXYT -> XYZ : XYZ (7x I128)

void pointPairXYTZ_loadXYT(PointPairXYTZ_t* pair, I128* p1, I128* p2);                        // load XYT : XYT (7x I128) -> PointPairXYTZ (Z initialed to 1)
void pointPairXYTZ_loadXYTZ(PointPairXYTZ_t* pair, I128* p1, I128* p2);                       // load XYTZ : XYTZ (10x I128) -> PointPairXYTZ
void pointPairXYTZ_loadH_XYTZ(PointPairXYTZ_t* pair, PointPairXYTZ_t* acc, I128* pt);         // load acc.H : XYTZ (10x I128) -> PointPairXYTZ
void pointPairXYTZ_store(I128* p1, I128* p2, PointPairXYTZ_t* pair);                          // store PointPairXYTZ -> XYTZ : XYTZ (10x I128)

void pointPairXYTZ_accumulateXYT_unified(PointPairXYTZ_t* acc, PointPairXYT_t* pt);
void pointPairXYTZ_accumulateXYTZ_unified(PointPairXYTZ_t* acc, PointPairXYTZ_t* pt);
void pointPairXYTZ_accumulateXYT(PointPairXYTZ_t* acc, PointPairXYT_t* pt);
void pointPairXYTZ_accumulateXYTZ(PointPairXYTZ_t* acc, PointPairXYTZ_t* pt);
