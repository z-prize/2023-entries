/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

void fieldPairLoad(FieldPair_t* fp, I128* f1, I128* f2);
void fieldPairExtractLL(FieldPair_t* r, FieldPair_t* f1, FieldPair_t* f2);
void fieldPairExtractHH(FieldPair_t* r, FieldPair_t* f1, FieldPair_t* f2);

void fieldPairConvert(FieldPair_t* r, FieldPair_t* fp);
void fieldPairResolve(FieldPair_t* r, FieldPair_t* fp);
void fieldPairResolveConvert(FieldPair_t* r, FieldPair_t* fp);
void fieldPairReduce(FieldPair_t* r, FieldPair_t* fp);
void fieldPairReduceResolve(FieldPair_t* r, FieldPair_t* fp);
void fieldPairFullReduceResolve(FieldPair_t* r, FieldPair_t* fp);
void fieldPairAdd(FieldPair_t* r, FieldPair_t* a, FieldPair_t* b);
void fieldPairSub(FieldPair_t* r, FieldPair_t* a, FieldPair_t* b);
void fieldPairMul(FieldPair_t* r, FieldPair_t* a, FieldPair_t* b);
void fieldPairConvertMul(FieldPair_t* r, FieldPair_t* a, FieldPair_t* b);
void fieldPairResolveConvertMul(FieldPair_t* r, FieldPair_t* a, FieldPair_t* b);


#if defined(X86)
void fieldPairDumpL(FieldPair_t* fp);
void fieldPairDumpH(FieldPair_t* fp);
#endif
