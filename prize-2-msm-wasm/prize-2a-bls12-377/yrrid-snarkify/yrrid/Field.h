/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

void fieldLoad(Field_t* f, I128* data);
void fieldStore(I128* data, Field_t* f);
void fieldResolve(Field_t* f);
void fieldPartialResolve(Field_t* f);
void fieldAdd(Field_t* r, Field_t* a, Field_t* b);
void fieldSub(Field_t* r, Field_t* a, Field_t* b);
void fieldMul(Field_t* r0, Field_t* r1, Field_t* a0, Field_t* b0, Field_t* a1, Field_t* b1);
void fieldInv(Field_t* r, Field_t* f);

bool fieldIsZero(Field_t* f);
void fieldReduce(Field_t* f);
void fieldSetZero(Field_t* f);
void fieldSetOne(Field_t* f) ;
void fieldSetPhi(Field_t* f) ;
void fieldSetN(Field_t* f);
void fieldSet2N(Field_t* f);
void fieldSet3N(Field_t* f);
void fieldSet4N(Field_t* f);
void fieldSet5N(Field_t* f);
void fieldSet6N(Field_t* f);
void fieldSetRInverse(Field_t* f);
void fieldSetRSquared(Field_t* f);

