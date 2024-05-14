/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

#pragma once

namespace Host {

// Host::Curve<Host::BLS12377G1>::PointXY.

template<class FieldClass>
class Curve {
  public:
  typedef typename FieldClass::Field Field;

  class PointXY;
  class PointXYZZ;
  class PointXYT;
  class PointXYTZ;

  class PointXY {
    public:
    Field x, y;

    PointXY();
    PointXY(const Field& inX, const Field& inY);
    static PointXY infinity();
    static PointXY generator();
    static PointXY fromCString(const char* cString);
    void load(const uint64_t* serializedWords);
    void store(uint64_t* serializedWords) const;
    void fromHostMontgomeryToDeviceMontgomery();
    void fromDeviceMontgomeryToHostMontgomery();
    void toHostMontgomery();
    void fromHostMontgomery();
    void toDeviceMontgomery();
    void fromDeviceMontgomery();

    void      negate();
    bool      isInfinity() const;
    PointXYZZ scale(const uint32_t amount) const;
    PointXYZZ scale(const uint8_t* bytes, const uint32_t byteCount) const;

    void      dump(const bool convertFromHostMontgomery=true) const;
  };

  class PointXYZZ {
    public:
    Field x, y, zz, zzz;

    PointXYZZ();
    PointXYZZ(const Field& x, const Field& y, const Field& zz, const Field& zzz);
    static PointXYZZ infinity();
    static PointXYZZ fromCString(const char* cString);
    void load(const uint64_t* serializedWords);
    void store(uint64_t* serializedWords) const;
    void fromHostMontgomeryToDeviceMontgomery();
    void fromDeviceMontgomeryToHostMontgomery();
    void toHostMontgomery();
    void fromHostMontgomery();
    void toDeviceMontgomery();
    void fromDeviceMontgomery();

    void negate();
    void dbl();
    void addXY(const PointXY& point);
    void addXYZZ(const PointXYZZ& point);

    bool      isInfinity() const;
    PointXY   normalize() const;
    PointXYZZ scale(const uint32_t amount) const;
    PointXYZZ scale(const uint8_t* bytes, const uint32_t byteCount) const;

    void      dump(const bool convertFromHostMontgomery=true) const;
  };

  class PointXYTZ {
    public:
    Field x, y, t, z;

    PointXYTZ();
    PointXYTZ(const Field& x, const Field& y, const Field& t, const Field& z);
    PointXYTZ(const PointXY& point);
    static PointXYTZ infinity();
    static PointXYTZ generator();
    static PointXYTZ fromCString(const char* cString);
    void load(const uint64_t* serializedWords);
    void store(uint64_t* serializedWords) const;
    void fromHostMontgomeryToDeviceMontgomery();
    void fromDeviceMontgomeryToHostMontgomery();
    void toHostMontgomery();
    void fromHostMontgomery();
    void toDeviceMontgomery();
    void fromDeviceMontgomery();

    void      negate();
    void      dbl();
    void      addXYTZ(const PointXYTZ& point);

    bool      isInfinity() const;
    PointXY   normalize() const;
    PointXYTZ scale(const uint32_t amount) const;
    PointXYTZ scale(const uint8_t* bytes, const uint32_t byteCount) const;

    void      dump(const bool convertFromHostMontgomery=true) const;
  };
};

const char* skipDelimiters(const char* cString) {
  char next=*cString;

  while(next==' ' || next=='\n' || next=='\r' || next=='\t') {
    cString++;
    next=*cString;
  }
  return cString;
}

const char* skipToEndOfHex(const char* cString) {
  char        next=*cString;

  while((next>='0' && next<='9') || (next>='a' && next<='f') || (next>='A' && next<='F')) {
    cString++;
    next=*cString;
  }
  return cString;
}

int64_t nibble(char value) {
  if(value>='0' && value<='9')
    return value-'0';
  if(value>='a' && value<='f')
    return value-'a'+10;
  if(value>='A' && value<='F')
    return value-'A'+10;
  return -1;
}

const char* readHex(uint64_t* words, uint32_t wordCount, const char* cString) {
  const char *start=skipDelimiters(cString);
  const char *stop=skipToEndOfHex(start);
  const char *current;
  char        next=*stop;
  int64_t     insert;
  uint32_t    offset;

  for(uint32_t i=0;i<wordCount;i++)
    words[i]=0;

  if(next!=0 && next!=' ' && next!='\t' && next!='\n' && next!='\r') {
    fprintf(stderr, "Error in readHex() -- bad hex char in cString!\n");
    exit(1);
  }

  if(start==stop) {
    fprintf(stderr, "Error in readHex() -- no data found in cString!\n");
    exit(1);
  }

  current=stop-1;
  while(current>=start) {
    offset=stop-current-1;
    insert=nibble(*current);
    if(insert!=0 && offset>=wordCount*16) {
      fprintf(stderr, "Error in readHex() -- hex value too long!\n");
      exit(1);
    }
    if(insert!=0)
      words[offset/16]+=insert<<(offset%16)*4;
    current--;
  }
  return stop;
}


/*************************************************************************************************
 * POINT XY CLASS IMPLEMENTATION
 *************************************************************************************************/

template<class FieldClass>
Curve<FieldClass>::PointXY::PointXY() {
  FieldClass::setZero(x);
  FieldClass::setZero(y);
  x[6]=0x0020000000000000ull;
}

template<class FieldClass>
Curve<FieldClass>::PointXY::PointXY(const Field& inX, const Field& inY) {
  FieldClass::set(x, inX);
  FieldClass::set(y, inY);
}

template<class FieldClass>
typename Curve<FieldClass>::PointXY Curve<FieldClass>::PointXY::infinity() {
  return PointXY();
}

template<class FieldClass>
typename Curve<FieldClass>::PointXY Curve<FieldClass>::PointXY::generator() {
  PointXY pt;

  FieldClass::setGeneratorX(pt.x);
  FieldClass::setGeneratorY(pt.y);
  return pt;
}

template<class FieldClass>
typename Curve<FieldClass>::PointXY Curve<FieldClass>::PointXY::fromCString(const char* cString) {
  PointXY     pt;
  const char* start;
  uint64_t    serialized[12];

  cString=readHex(serialized, 6, cString);
  cString=readHex(serialized+6, 6, cString);
  pt.load(serialized);
  return pt;
}

template<class FieldClass>
void Curve<FieldClass>::PointXY::load(const uint64_t* serializedWords) {
  FieldClass::load(x, serializedWords+0);
  FieldClass::load(y, serializedWords+6);
}

template<class FieldClass>
void Curve<FieldClass>::PointXY::store(uint64_t* serializedWords) const {
  FieldClass::store(serializedWords+0, x);
  FieldClass::store(serializedWords+6, y);
}

template<class FieldClass>
void Curve<FieldClass>::PointXY::fromHostMontgomeryToDeviceMontgomery() {
  if(isInfinity()) return;
  FieldClass::fromHostMontgomeryToDeviceMontgomery(x, x);
  FieldClass::fromHostMontgomeryToDeviceMontgomery(y, y);
}

template<class FieldClass>
void Curve<FieldClass>::PointXY::fromDeviceMontgomeryToHostMontgomery() {
  if(isInfinity()) return;
  FieldClass::fromDeviceMontgomeryToHostMontgomery(x, x);
  FieldClass::fromDeviceMontgomeryToHostMontgomery(y, y);
}

template<class FieldClass>
void Curve<FieldClass>::PointXY::toHostMontgomery() {
  if(isInfinity()) return;
  FieldClass::toMontgomery(x, x);
  FieldClass::toMontgomery(y, y);
}

template<class FieldClass>
void Curve<FieldClass>::PointXY::fromHostMontgomery() {
  if(isInfinity()) return;
  FieldClass::fromMontgomery(x, x);
  FieldClass::fromMontgomery(y, y);
}

template<class FieldClass>
void Curve<FieldClass>::PointXY::toDeviceMontgomery() {
  if(isInfinity()) return;
  FieldClass::toMontgomery(x, x);
  FieldClass::toMontgomery(y, y);
  FieldClass::fromHostMontgomeryToDeviceMontgomery(x, x);
  FieldClass::fromHostMontgomeryToDeviceMontgomery(y, y);
}

template<class FieldClass>
void Curve<FieldClass>::PointXY::fromDeviceMontgomery() {
  if(isInfinity()) return;
  FieldClass::fromDeviceMontgomeryToHostMontgomery(x, x);
  FieldClass::fromDeviceMontgomeryToHostMontgomery(y, y);
  FieldClass::fromMontgomery(x, x);
  FieldClass::fromMontgomery(y, y);
}

template<class FieldClass>
void Curve<FieldClass>::PointXY::negate() {
  FieldClass::sub(y, FieldClass::Field::constN, y);
}

template<class FieldClass>
bool Curve<FieldClass>::PointXY::isInfinity() const {
  return (x[6] & 0x0020000000000000ull)!=0;
}

template<class FieldClass>
typename Curve<FieldClass>::PointXYZZ Curve<FieldClass>::PointXY::scale(const uint32_t amount) const {
  PointXYZZ acc;

  for(int i=31;i>=0;i--) {
    acc.dbl();
    if(((amount>>i) & 0x01)!=0)
      acc.addXY(*this);
  }
  return acc;
}

template<class FieldClass>
typename Curve<FieldClass>::PointXYZZ Curve<FieldClass>::PointXY::scale(const uint8_t* bytes, const uint32_t byteCount) const {
  PointXYZZ acc;

  for(int32_t i=byteCount*8-1;i>=0;i--) {
    acc.dbl();
    if(((bytes[i/8]>>i%8) & 0x01)!=0)
      acc.addXY(*this);
  }
  return acc;
}

template<class FieldClass>
void Curve<FieldClass>::PointXY::dump(bool convertFromHostMontgomery) const {
  typename FieldClass::Field local;
  uint64_t                   xWords[6], yWords[6];

  if(isInfinity()) {
    printf("Point at infinity\n");
    return;
  }

  if(convertFromHostMontgomery) {
    FieldClass::fromMontgomery(local, x);
    FieldClass::store(xWords, local);
    FieldClass::fromMontgomery(local, y);
    FieldClass::store(yWords, local);
  }
  else {
    FieldClass::store(xWords, x);
    FieldClass::store(yWords, y);
  }

  printf("  x = 0x%016lX%016lX%016lX%016lX%016lX%016lX\n", xWords[5], xWords[4], xWords[3], xWords[2], xWords[1], xWords[0]);
  printf("  y = 0x%016lX%016lX%016lX%016lX%016lX%016lX\n", yWords[5], yWords[4], yWords[3], yWords[2], yWords[1], yWords[0]);
}

/*************************************************************************************************
 * POINT XYZZ CLASS IMPLEMENTATION
 *************************************************************************************************/

template<class FieldClass>
Curve<FieldClass>::PointXYZZ::PointXYZZ() {
  FieldClass::setZero(x);
  FieldClass::setZero(y);
  FieldClass::setZero(zz);
  FieldClass::setZero(zzz);
}

template<class FieldClass>
Curve<FieldClass>::PointXYZZ::PointXYZZ(const Field& inX, const Field& inY, const Field& inZZ, const Field& inZZZ) {
  FieldClass::set(x, inX);
  FieldClass::set(y, inY);
  FieldClass::set(zz, inZZ);
  FieldClass::set(zzz, inZZZ);
}

template<class FieldClass>
typename Curve<FieldClass>::PointXYZZ Curve<FieldClass>::PointXYZZ::infinity() {
  return PointXYZZ();
}

template<class FieldClass>
typename Curve<FieldClass>::PointXYZZ Curve<FieldClass>::PointXYZZ::fromCString(const char* cString) {
  PointXYZZ   pt;
  const char* start;
  uint64_t    serialized[24];

  cString=readHex(serialized, 6, cString);
  cString=readHex(serialized+6, 6, cString);
  cString=readHex(serialized+12, 6, cString);
  cString=readHex(serialized+18, 6, cString);
  pt.load(serialized);
  return pt;
}

template<class FieldClass>
void Curve<FieldClass>::PointXYZZ::load(const uint64_t* serializedWords) {
  FieldClass::load(x, serializedWords+0);
  FieldClass::load(y, serializedWords+6);
  FieldClass::load(zz, serializedWords+12);
  FieldClass::load(zzz, serializedWords+18);
}

template<class FieldClass>
void Curve<FieldClass>::PointXYZZ::store(uint64_t* serializedWords) const {
  FieldClass::store(serializedWords+0, x);
  FieldClass::store(serializedWords+6, y);
  FieldClass::store(serializedWords+12, zz);
  FieldClass::store(serializedWords+18, zzz);
}

template<class FieldClass>
void Curve<FieldClass>::PointXYZZ::fromHostMontgomeryToDeviceMontgomery() {
  FieldClass::fromHostMontgomeryToDeviceMontgomery(x, x);
  FieldClass::fromHostMontgomeryToDeviceMontgomery(y, y);
  FieldClass::fromHostMontgomeryToDeviceMontgomery(zz, zz);
  FieldClass::fromHostMontgomeryToDeviceMontgomery(zzz, zzz);
}

template<class FieldClass>
void Curve<FieldClass>::PointXYZZ::fromDeviceMontgomeryToHostMontgomery() {
  FieldClass::fromDeviceMontgomeryToHostMontgomery(x, x);
  FieldClass::fromDeviceMontgomeryToHostMontgomery(y, y);
  FieldClass::fromDeviceMontgomeryToHostMontgomery(zz, zz);
  FieldClass::fromDeviceMontgomeryToHostMontgomery(zzz, zzz);
}

template<class FieldClass>
void Curve<FieldClass>::PointXYZZ::toHostMontgomery() {
  FieldClass::toMontgomery(x, x);
  FieldClass::toMontgomery(y, y);
  FieldClass::toMontgomery(zz, zz);
  FieldClass::toMontgomery(zzz, zzz);
}

template<class FieldClass>
void Curve<FieldClass>::PointXYZZ::fromHostMontgomery() {
  FieldClass::fromMontgomery(x, x);
  FieldClass::fromMontgomery(y, y);
  FieldClass::fromMontgomery(zz, zz);
  FieldClass::fromMontgomery(zzz, zzz);
}

template<class FieldClass>
void Curve<FieldClass>::PointXYZZ::toDeviceMontgomery() {
  FieldClass::toMontgomery(x, x);
  FieldClass::toMontgomery(y, y);
  FieldClass::toMontgomery(zz, zz);
  FieldClass::toMontgomery(zzz, zzz);
  FieldClass::fromHostMontgomeryToDeviceMontgomery(x, x);
  FieldClass::fromHostMontgomeryToDeviceMontgomery(y, y);
  FieldClass::fromHostMontgomeryToDeviceMontgomery(zz, zz);
  FieldClass::fromHostMontgomeryToDeviceMontgomery(zzz, zzz);
}

template<class FieldClass>
void Curve<FieldClass>::PointXYZZ::fromDeviceMontgomery() {
  FieldClass::fromDeviceMontgomeryToHostMontgomery(x, x);
  FieldClass::fromDeviceMontgomeryToHostMontgomery(y, y);
  FieldClass::fromDeviceMontgomeryToHostMontgomery(zz, zz);
  FieldClass::fromDeviceMontgomeryToHostMontgomery(zzz, zzz);
  FieldClass::fromMontgomery(x, x);
  FieldClass::fromMontgomery(y, y);
  FieldClass::fromMontgomery(zz, zz);
  FieldClass::fromMontgomery(zzz, zzz);
}

template<class FieldClass>
void Curve<FieldClass>::PointXYZZ::negate() {
  FieldClass::sub(y, FieldClass::Field::constN, y);
}

template<class FieldClass>
void Curve<FieldClass>::PointXYZZ::dbl() {
  typename FieldClass::Field U, V, W, S, M, T, X, Y;

  if(FieldClass::isZero(zz))
    return;

  FieldClass::add(U, y, y);
  FieldClass::mul(V, U, U);
  FieldClass::mul(W, U, V);
  FieldClass::mul(S, x, V);
  FieldClass::mul(M, x, x);
  FieldClass::add(T, M, M);
  FieldClass::add(M, T, M);
  FieldClass::mul(X, M, M);
  FieldClass::sub(X, X, S);
  FieldClass::sub(X, X, S);
  FieldClass::sub(T, S, X);
  FieldClass::mul(Y, M, T);
  FieldClass::mul(T, W, y);
  FieldClass::sub(Y, Y, T);
  FieldClass::set(x, X);
  FieldClass::set(y, Y);
  FieldClass::mul(zz, zz, V);
  FieldClass::mul(zzz, zzz, W);
}

template<class FieldClass>
void Curve<FieldClass>::PointXYZZ::addXY(const PointXY& point) {
  typename FieldClass::Field U2, S2, P, R, PP, PPP, Q, T, X, Y;

  if(point.isInfinity())
    return;

  if(FieldClass::isZero(zz)) {
    FieldClass::set(x, point.x);
    FieldClass::set(y, point.y);
    FieldClass::setOne(zz);
    FieldClass::setOne(zzz);
    return;
  }

  FieldClass::mul(U2, point.x, zz);
  FieldClass::mul(S2, point.y, zzz);
  FieldClass::sub(P, U2, x);
  FieldClass::sub(R, S2, y);

  if(FieldClass::isZero(P) && FieldClass::isZero(R)) {
    dbl();
    return;
  }

  FieldClass::mul(PP, P, P);
  FieldClass::mul(PPP, PP, P);
  FieldClass::mul(Q, x, PP);
  FieldClass::mul(X, R, R);
  FieldClass::sub(X, X, PPP);
  FieldClass::sub(X, X, Q);
  FieldClass::sub(X, X, Q);
  FieldClass::sub(T, Q, X);
  FieldClass::mul(Y, R, T);
  FieldClass::mul(T, y, PPP);
  FieldClass::sub(Y, Y, T);
  FieldClass::set(x, X);
  FieldClass::set(y, Y);
  FieldClass::mul(zz, zz, PP);
  FieldClass::mul(zzz, zzz, PPP);
}

template<class FieldClass>
void Curve<FieldClass>::PointXYZZ::addXYZZ(const PointXYZZ& point) {
  typename FieldClass::Field U1, U2, S1, S2, P, R, PP, PPP, Q, T, X, Y;

  if(point.isInfinity())
    return;

  if(FieldClass::isZero(zz)) {
    FieldClass::set(x, point.x);
    FieldClass::set(y, point.y);
    FieldClass::set(zz, point.zz);
    FieldClass::set(zzz, point.zzz);
    return;
  }

  FieldClass::mul(U1, x, point.zz);
  FieldClass::mul(U2, point.x, zz);
  FieldClass::mul(S1, y, point.zzz);
  FieldClass::mul(S2, point.y, zzz);
  FieldClass::sub(P, U2, U1);
  FieldClass::sub(R, S2, S1);
  if(FieldClass::isZero(P) && FieldClass::isZero(R)) {
    dbl();
    return;
  }

  FieldClass::mul(PP, P, P);
  FieldClass::mul(PPP, PP, P);
  FieldClass::mul(Q, U1, PP);
  FieldClass::mul(X, R, R);
  FieldClass::sub(X, X, PPP);
  FieldClass::sub(X, X, Q);
  FieldClass::sub(X, X, Q);
  FieldClass::sub(T, Q, X);
  FieldClass::mul(Y, R, T);
  FieldClass::mul(T, S1, PPP);
  FieldClass::sub(Y, Y, T);
  FieldClass::set(x, X);
  FieldClass::set(y, Y);
  FieldClass::mul(zz, zz, point.zz);
  FieldClass::mul(zzz, zzz, point.zzz);
  FieldClass::mul(zz, zz, PP);
  FieldClass::mul(zzz, zzz, PPP);
}

template<class FieldClass>
bool Curve<FieldClass>::PointXYZZ::isInfinity() const {
  return FieldClass::isZero(zz);
}

template<class FieldClass>
typename Curve<FieldClass>::PointXY Curve<FieldClass>::PointXYZZ::normalize() const {
  typename FieldClass::Field iii, ii, i, xNormalized, yNormalized;

  if(FieldClass::isZero(zz))
    return PointXY();

  FieldClass::inv(iii, zzz);
  FieldClass::mul(yNormalized, y, iii);
  FieldClass::mul(i, iii, zz);
  FieldClass::sqr(ii, i);
  FieldClass::mul(xNormalized, x, ii);
  return PointXY(xNormalized, yNormalized);
}

template<class FieldClass>
typename Curve<FieldClass>::PointXYZZ Curve<FieldClass>::PointXYZZ::scale(const uint32_t amount) const {
  PointXYZZ acc;

  for(int i=31;i>=0;i--) {
    acc.dbl();
    if(((amount>>i) & 0x01)!=0)
      acc.addXYZZ(*this);
  }
  return acc;
}

template<class FieldClass>
typename Curve<FieldClass>::PointXYZZ Curve<FieldClass>::PointXYZZ::scale(const uint8_t* bytes, const uint32_t byteCount) const {
  PointXYZZ acc;

  for(int32_t i=byteCount*8-1;i>=0;i--) {
    acc.dbl();
    if(((bytes[i/8]>>i%8) & 0x01)!=0)
      acc.addXYZZ(*this);
  }
  return acc;
}

template<class FieldClass>
void Curve<FieldClass>::PointXYZZ::dump(bool convertFromHostMontgomery) const {
  typename FieldClass::Field local;
  uint64_t                   xWords[6], yWords[6], zzWords[6], zzzWords[6];

  if(FieldClass::isZero(zz)) {
    printf("Point at infinity\n");
    return;
  }

  if(convertFromHostMontgomery) {
    FieldClass::fromMontgomery(local, x);
    FieldClass::store(xWords, local);
    FieldClass::fromMontgomery(local, y);
    FieldClass::store(yWords, local);
    FieldClass::fromMontgomery(local, zz);
    FieldClass::store(zzWords, local);
    FieldClass::fromMontgomery(local, zzz);
    FieldClass::store(zzzWords, local);
  }
  else {
    FieldClass::store(xWords, x);
    FieldClass::store(yWords, y);
    FieldClass::store(zzWords, zz);
    FieldClass::store(zzzWords, zzz);
  }

  printf("  x = 0x%016lX%016lX%016lX%016lX%016lX%016lX\n", xWords[5], xWords[4], xWords[3], xWords[2], xWords[1], xWords[0]);
  printf("  y = 0x%016lX%016lX%016lX%016lX%016lX%016lX\n", yWords[5], yWords[4], yWords[3], yWords[2], yWords[1], yWords[0]);
  printf(" zz = 0x%016lX%016lX%016lX%016lX%016lX%016lX\n", zzWords[5], zzWords[4], zzWords[3], zzWords[2], zzWords[1], zzWords[0]);
  printf("zzz = 0x%016lX%016lX%016lX%016lX%016lX%016lX\n", zzzWords[5], zzzWords[4], zzzWords[3], zzzWords[2], zzzWords[1], zzzWords[0]);
}

/*************************************************************************************************
 * POINT XYTZ CLASS IMPLEMENTATION
 *************************************************************************************************/

template<class FieldClass>
Curve<FieldClass>::PointXYTZ::PointXYTZ() {
  FieldClass::setZero(x);
  FieldClass::setOne(y);
  FieldClass::setZero(t);
  FieldClass::setOne(z);
}

template<class FieldClass>
Curve<FieldClass>::PointXYTZ::PointXYTZ(const Field& inX, const Field& inY, const Field& inT, const Field& inZ) {
  FieldClass::set(x, inX);
  FieldClass::set(y, inY);
  FieldClass::set(t, inT);
  FieldClass::set(z, inZ);
}

template<class FieldClass>
Curve<FieldClass>::PointXYTZ::PointXYTZ(const PointXY& point) {
  typename FieldClass::Field one, i, f, s, t0, t1;

  FieldClass::setOne(one);
  FieldClass::setTwistedEdwardsF(f);
  FieldClass::setTwistedEdwardsS(s);

  FieldClass::add(t0, point.x, one);   // t0=x+1
  FieldClass::mul(x, t0, f);
  FieldClass::inv(i, point.y);
  FieldClass::mul(x, x, i);            // x=f*(pt.x+1)/pt.y

  FieldClass::mul(t0, s, t0);          // t0=s*x+s
  FieldClass::sub(t1, t0, one);        // t1=s*x+s-1

  FieldClass::add(t0, t0, one);        // t0=s*x+s+1
  FieldClass::inv(i, t0);
  FieldClass::mul(y, t1, i);           // y=(s*x+s-1)/(sx+s+1)

  FieldClass::mul(t, x, y);

  FieldClass::setOne(z);
}

template<class FieldClass>
typename Curve<FieldClass>::PointXYTZ Curve<FieldClass>::PointXYTZ::infinity() {
  return PointXYTZ();
}

template<class FieldClass>
typename Curve<FieldClass>::PointXYTZ Curve<FieldClass>::PointXYTZ::generator() {
  PointXYTZ pt;

  FieldClass::setTwistedEdwardsGeneratorX(pt.x);
  FieldClass::setTwistedEdwardsGeneratorY(pt.y);
  FieldClass::setTwistedEdwardsGeneratorT(pt.t);
  FieldClass::setOne(pt.z);
  return pt;
}

template<class FieldClass>
typename Curve<FieldClass>::PointXYTZ Curve<FieldClass>::PointXYTZ::fromCString(const char* cString) {
  PointXYZZ   pt;
  const char* start;
  uint64_t    serialized[24];

  cString=readHex(serialized, 6, cString);
  cString=readHex(serialized+6, 6, cString);
  cString=readHex(serialized+12, 6, cString);
  cString=readHex(serialized+18, 6, cString);
  pt.load(serialized);
  return pt;
}

template<class FieldClass>
void Curve<FieldClass>::PointXYTZ::load(const uint64_t* serializedWords) {
  FieldClass::load(x, serializedWords+0);
  FieldClass::load(y, serializedWords+6);
  FieldClass::load(t, serializedWords+12);
  FieldClass::load(z, serializedWords+18);
}

template<class FieldClass>
void Curve<FieldClass>::PointXYTZ::store(uint64_t* serializedWords) const {
  FieldClass::store(serializedWords+0, x);
  FieldClass::store(serializedWords+6, y);
  FieldClass::store(serializedWords+12, t);
  FieldClass::store(serializedWords+18, z);
}

template<class FieldClass>
void Curve<FieldClass>::PointXYTZ::fromHostMontgomeryToDeviceMontgomery() {
  FieldClass::fromHostMontgomeryToDeviceMontgomery(x, x);
  FieldClass::fromHostMontgomeryToDeviceMontgomery(y, y);
  FieldClass::fromHostMontgomeryToDeviceMontgomery(t, t);
  FieldClass::fromHostMontgomeryToDeviceMontgomery(z, z);
}

template<class FieldClass>
void Curve<FieldClass>::PointXYTZ::fromDeviceMontgomeryToHostMontgomery() {
  FieldClass::fromDeviceMontgomeryToHostMontgomery(x, x);
  FieldClass::fromDeviceMontgomeryToHostMontgomery(y, y);
  FieldClass::fromDeviceMontgomeryToHostMontgomery(t, t);
  FieldClass::fromDeviceMontgomeryToHostMontgomery(z, z);
}

template<class FieldClass>
void Curve<FieldClass>::PointXYTZ::toHostMontgomery() {
  FieldClass::toMontgomery(x, x);
  FieldClass::toMontgomery(y, y);
  FieldClass::toMontgomery(t, t);
  FieldClass::toMontgomery(z, z);
}

template<class FieldClass>
void Curve<FieldClass>::PointXYTZ::fromHostMontgomery() {
  FieldClass::fromMontgomery(x, x);
  FieldClass::fromMontgomery(y, y);
  FieldClass::fromMontgomery(t, t);
  FieldClass::fromMontgomery(z, z);
}

template<class FieldClass>
void Curve<FieldClass>::PointXYTZ::toDeviceMontgomery() {
  FieldClass::toMontgomery(x, x);
  FieldClass::toMontgomery(y, y);
  FieldClass::toMontgomery(t, t);
  FieldClass::toMontgomery(z, z);
  FieldClass::fromHostMontgomeryToDeviceMontgomery(x, x);
  FieldClass::fromHostMontgomeryToDeviceMontgomery(y, y);
  FieldClass::fromHostMontgomeryToDeviceMontgomery(t, t);
  FieldClass::fromHostMontgomeryToDeviceMontgomery(z, z);
}

template<class FieldClass>
void Curve<FieldClass>::PointXYTZ::fromDeviceMontgomery() {
  FieldClass::fromDeviceMontgomeryToHostMontgomery(x, x);
  FieldClass::fromDeviceMontgomeryToHostMontgomery(y, y);
  FieldClass::fromDeviceMontgomeryToHostMontgomery(t, t);
  FieldClass::fromDeviceMontgomeryToHostMontgomery(z, z);
  FieldClass::fromMontgomery(x, x);
  FieldClass::fromMontgomery(y, y);
  FieldClass::fromMontgomery(t, t);
  FieldClass::fromMontgomery(z, z);
}

template<class FieldClass>
void Curve<FieldClass>::PointXYTZ::negate() {
  FieldClass::sub(x, FieldClass::Field::constN, x);
  FieldClass::sub(t, FieldClass::Field::constN, t);
}

template<class FieldClass>
void Curve<FieldClass>::PointXYTZ::dbl() {
   add(*this);
}

template<class FieldClass>
void Curve<FieldClass>::PointXYTZ::addXYTZ(const PointXYTZ& point) {
  typename FieldClass::Field A, B, C, D, E, F, G, H, T0, T1, K;

  FieldClass::setTwistedEdwardsK(K);

  FieldClass::sub(T0, y, x);
  FieldClass::sub(T1, point.y, point.x);
  FieldClass::mul(A, T0, T1);
  FieldClass::add(T0, y, x);
  FieldClass::add(T1, point.y, point.x);
  FieldClass::mul(B, T0, T1);
  FieldClass::mul(C, t, point.t);
  FieldClass::mul(C, C, K);
  FieldClass::mul(D, z, point.z);
  FieldClass::add(D, D, D);
  FieldClass::sub(E, B, A);
  FieldClass::sub(F, D, C);
  FieldClass::add(G, D, C);
  FieldClass::add(H, B, A);
  FieldClass::mul(x, E, F);
  FieldClass::mul(y, G, H);
  FieldClass::mul(t, E, H);
  FieldClass::mul(z, F, G);
}

template<class FieldClass>
bool Curve<FieldClass>::PointXYTZ::isInfinity() const {
  return FieldClass::isZero(x);
}

template<class FieldClass>
typename Curve<FieldClass>::PointXY Curve<FieldClass>::PointXYTZ::normalize() const {
  typename FieldClass::Field i, ptX, ptY, one, f, bInv, onePlus, oneMinus, temp;

  if(FieldClass::isZero(x))
    return PointXY();

  FieldClass::setOne(one);
  FieldClass::setTwistedEdwardsF(f);
  FieldClass::setTwistedEdwardsBInv(bInv);

  FieldClass::inv(i, z);
  FieldClass::mul(ptX, x, i);
  FieldClass::mul(ptY, y, i);

  FieldClass::add(onePlus, one, ptY);
  FieldClass::sub(oneMinus, one, ptY);
  FieldClass::mul(temp, ptX, oneMinus);
  FieldClass::inv(i, temp);
  FieldClass::mul(temp, f, onePlus);
  FieldClass::mul(ptY, temp, i);
  FieldClass::mul(ptY, ptY, bInv);

  FieldClass::inv(i, oneMinus);
  FieldClass::mul(ptX, onePlus, i);
  FieldClass::mul(ptX, ptX, bInv);
  FieldClass::sub(ptX, ptX, one);

  return PointXY(ptX, ptY);
}

template<class FieldClass>
typename Curve<FieldClass>::PointXYTZ Curve<FieldClass>::PointXYTZ::scale(const uint32_t amount) const {
  PointXYTZ acc;

  for(int i=31;i>=0;i--) {
    acc.addXYTZ(acc);
    if(((amount>>i) & 0x01)!=0)
      acc.addXYTZ(*this);
  }
  return acc;
}

template<class FieldClass>
typename Curve<FieldClass>::PointXYTZ Curve<FieldClass>::PointXYTZ::scale(const uint8_t* bytes, const uint32_t byteCount) const {
  PointXYTZ acc;

  for(int32_t i=byteCount*8-1;i>=0;i--) {
    acc.addXYTZ(acc);
    if(((bytes[i/8]>>i%8) & 0x01)!=0)
      acc.addXYTZ(*this);
  }
  return acc;
}

template<class FieldClass>
void Curve<FieldClass>::PointXYTZ::dump(bool convertFromHostMontgomery) const {
  typename FieldClass::Field local;
  uint64_t                   xWords[6], yWords[6], tWords[6], zWords[6];

  if(FieldClass::isZero(x)) {
    printf("Point at infinity\n");
    return;
  }

  if(convertFromHostMontgomery) {
    FieldClass::fromMontgomery(local, x);
    FieldClass::store(xWords, local);
    FieldClass::fromMontgomery(local, y);
    FieldClass::store(yWords, local);
    FieldClass::fromMontgomery(local, t);
    FieldClass::store(tWords, local);
    FieldClass::fromMontgomery(local, z);
    FieldClass::store(zWords, local);
  }
  else {
    FieldClass::store(xWords, x);
    FieldClass::store(yWords, y);
    FieldClass::store(tWords, t);
    FieldClass::store(zWords, z);
  }

  printf("  x = 0x%016lX%016lX%016lX%016lX%016lX%016lX\n", xWords[5], xWords[4], xWords[3], xWords[2], xWords[1], xWords[0]);
  printf("  y = 0x%016lX%016lX%016lX%016lX%016lX%016lX\n", yWords[5], yWords[4], yWords[3], yWords[2], yWords[1], yWords[0]);
  printf("  t = 0x%016lX%016lX%016lX%016lX%016lX%016lX\n", tWords[5], tWords[4], tWords[3], tWords[2], tWords[1], tWords[0]);
  printf("  z = 0x%016lX%016lX%016lX%016lX%016lX%016lX\n", zWords[5], zWords[4], zWords[3], zWords[2], zWords[1], zWords[0]);
}

} /* namespace Host */