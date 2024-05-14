template<typename Curve, bool montgomery>
class PointXY {
  public:
  typedef FF<typename Curve::OrderField,false>     OFF;
  typedef FF<typename Curve::MainField,montgomery> FF;

  bool                      infinity;
  typename Curve::MainField x, y;

  PointXY() {
    infinity=true;
  }

  PointXY(const typename Curve::MainField& inX, const typename Curve::MainField& inY) {
    infinity=false;
    x=inX;
    y=inY;
  }

  bool isInfinity() {
    return infinity;
  }

  PointXY negate() {
    return negate(*this);
  }

  PointXY dbl() {
    return dbl(*this);
  }

  PointXY add(const PointXY& pt) {
    return add(*this, pt);
  }

  PointXY scale(const typename Curve::OrderField& scalar) {
    return scale(*this, scalar);
  }

  static PointXY generator() {
    typename Curve::MainField generatorX(Curve::MainField::g1GeneratorX);
    typename Curve::MainField generatorY(Curve::MainField::g1GeneratorY);

    if(montgomery)
      return PointXY(FF::toMontgomery(generatorX), FF::toMontgomery(generatorY));
    return PointXY(generatorX, generatorY);
  }

  static PointXY random() {
    return Random::randomPoint<Curve,montgomery>();
  }

  static PointXY negate(const PointXY& pt) {
    if(pt.infinity)
      return pt;
    return PointXY(pt.x, FF::negate(pt.y));
  }

  static PointXY dbl(const PointXY& pt) {
    typename Curve::MainField lambda, newX, newY;;

    if(pt.infinity) return pt;

    lambda=FF::sqr(pt.x);
    lambda=FF::add(lambda, FF::add(lambda, lambda));
    lambda=FF::mul(lambda, FF::inv(FF::add(pt.y, pt.y)));
    newX=FF::sub(FF::sqr(lambda), FF::add(pt.x, pt.x));
    newY=FF::sub(FF::mul(lambda, FF::sub(pt.x, newX)), pt.y);
    return PointXY(newX, newY);
  }

  static PointXY add(const PointXY& pt1, const PointXY& pt2) {
    typename Curve::MainField dx, dy, lambda, newX, newY;

    if(pt1.infinity) return pt2;
    if(pt2.infinity) return pt1;

    dx=FF::sub(pt1.x, pt2.x);
    dy=FF::sub(pt1.y, pt2.y);
    if(FF::isZero(dx)) {
      if(FF::isZero(dy))
        return dbl(pt1);
      else
        return PointXY();
    }
    lambda=FF::mul(dy, FF::inv(dx));
    newX=FF::sub(FF::sqr(lambda), FF::add(pt1.x, pt2.x));
    newY=FF::sub(FF::mul(lambda, FF::sub(pt2.x, newX)), pt2.y);
    return PointXY(newX, newY);
  }

  static PointXY scale(const PointXY& pt, const typename Curve::OrderField& scalar) {
    PointXY acc;

    if(pt.infinity) return pt;

    for(int i=(Curve::OrderField::bits+63)/64*64-1;i>=0;i--) {
      acc=acc.dbl();
      if(OFF::testBit(scalar, i))
        acc=acc.add(pt);
    }
    return acc;
  }
};

template<typename Curve>
class PointXYZ {
  public:
  typename Curve::MainField::Words X, Y, Z;
};

