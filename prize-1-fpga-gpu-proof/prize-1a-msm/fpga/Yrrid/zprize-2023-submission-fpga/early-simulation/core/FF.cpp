// FIX FIX FIX - not multi-threaded!!!

template<typename Field, bool montgomery>
class FF {
  public:
  static void *ff;

  mpz_t zprime, znp, zmask, za, zb, zq, zr;

  FF() {
    mpz_inits(zprime, znp, zmask, za, zb, zq, zr, NULL);
    importWords(zprime, Field::prime);
    importWords(znp, Field::np);
    mpz_set_ui(zmask, 1);
    mpz_mul_2exp(zmask, zmask, (Field::bits+63)/64*64);
    mpz_sub_ui(zmask, zmask, 1);
  }

  ~FF() {
    mpz_clears(zprime, znp, zmask, za, zb, zq, zr, NULL);
  }

  void importWords(mpz_t x, const uint64_t* src) {
    mpz_import(x, (Field::bits+63)/64, -1, 8, 0, 0, src);
  }

  void exportWords(uint64_t* dst, mpz_t x) {
    uint32_t count;

    if(mpz_sizeinbase(x, 2)>=(Field::bits+63)/64*64) {
      printf("Houston, we have a problem\n");
      exit(1);
    }

    for(int i=0;i<(Field::bits+63)/64;i++)
      dst[i]=0;
    mpz_export(dst, NULL, -1, 8, 0, 0, x);
  }

  bool _isZero(const typename Field::Words& x) {
    uint64_t lor=x[0];

    // NOTE, does not detect multiples of p.  Use reduce if necessary.

    for(int i=1;i<(Field::bits+63)/64;i++)
      lor=lor | x[i];
    return lor==0;
  }

  bool _isEqual(const typename Field::Words& a, const typename Field::Words& b) {
    for(int i=0;i<(Field::bits+63)/64;i++)
      if(a[i]!=b[i])
        return false;
    return true;
  }

  void _negate(typename Field::Words& r, const typename Field::Words& x) {
    importWords(zr, x);
    if(mpz_sgn(zr)!=0)
      mpz_sub(zr, zprime, zr);
    exportWords(r, zr);
  }

  void _add(typename Field::Words& r, const typename Field::Words& a, const typename Field::Words& b) {
    importWords(za, a);
    importWords(zb, b);
    mpz_add(zr, za, zb);
    if(mpz_cmp(zr, zprime)>=0)
      mpz_sub(zr, zr, zprime);
    exportWords(r, zr);
  }

  void _sub(typename Field::Words& r, const typename Field::Words& a, const typename Field::Words& b) {
    importWords(za, a);
    importWords(zb, b);
    mpz_sub(zr, za, zb);
    if(mpz_sgn(zr)<0)
      mpz_add(zr, zr, zprime);
    exportWords(r, zr);
  }

  void _mul(typename Field::Words& r, const typename Field::Words& a, const typename Field::Words& b) {
    importWords(za, a);
    importWords(zb, b);
    mpz_mul(zr, za, zb);
    if(montgomery) {
      mpz_and(zq, zr, zmask);
      mpz_mul(zq, zq, znp);
      mpz_and(zq, zq, zmask);
      mpz_addmul(zr, zq, zprime);
      mpz_tdiv_q_2exp(zr, zr, (Field::bits+63)/64*64);
      if(mpz_cmp(zr, zprime)>=0)
        mpz_sub(zr, zr, zprime);
    }
    else
      mpz_mod(zr, zr, zprime);
    exportWords(r, zr);
  }

  void _inv(typename Field::Words& r, const typename Field::Words& x) {
    importWords(zr, x);
    mpz_invert(zr, zr, zprime);
    exportWords(r, zr);
    if(montgomery)
      _mul(r, r, Field::rCubedModP);
  }

  void _reduce(typename Field::Words& r, const typename Field::Words& x) {
    importWords(zr, x);
    mpz_mod(zr, zr, zprime);
    exportWords(r, zr);
  }

  void _reduce(typename Field::Words& r, const uint64_t* words, uint32_t wordCount) {
    mpz_import(zr, (Field::bits+63)/64, -1, 8, 0, 0, words);
    mpz_mod(zr, zr, zprime);
    exportWords(r, zr);
  }

  void _toMontgomery(typename Field::Words& r, const typename Field::Words& x) {
    importWords(zr, x);
    mpz_mul_2exp(zr, zr, (Field::bits+63)/64*64);
    mpz_mod(zr, zr, zprime);
    exportWords(r, zr);
  }

  void _fromMontgomery(typename Field::Words& r, const typename Field::Words& x) {
    importWords(zr, x);
    mpz_mul(zq, zr, znp);
    mpz_and(zq, zq, zmask);
    mpz_addmul(zr, zq, zprime);
    mpz_tdiv_q_2exp(zr, zr, (Field::bits+63)/64*64);
    if(mpz_cmp(zr, zprime)>=0)
      mpz_sub(zr, zr, zprime);
    exportWords(r, zr);
  }

  void _fromString(typename Field::Words& r, const char* cString) {
    mpz_set_str(zr, cString, 16);
    mpz_mod(zr, zr, zprime);
    exportWords(r, zr);
  }

  static bool isZero(const Field& x) {
    if(ff==NULL)
      ff=new FF();
    return ((FF*)ff)->_isZero(x.l);
  }

  static bool isEqual(const Field& a, const Field& b) {
    if(ff==NULL)
      ff=new FF();
    return ((FF*)ff)->_isEqual(a.l, b.l);
  }

  static bool testBit(const Field& x, uint32_t bit) {
    if(bit>=(Field::bits+63)/64*64)
      return false;
    return ((x.l[bit/64]>>(bit & 0x3F)) & 0x01)!=0;
  }

  static Field zero() {
    Field r;

    for(int i=0;i<(Field::bits+63)/64;i++)
      r.l[i]=0;
    return r;
  }

  static Field one() {
    Field r;

    r.l[0]=montgomery ? Field::rModP[0] : 1;
    for(int i=1;i<(Field::bits+63)/64;i++)
      r.l[i]=montgomery ? Field::rModP[i] : 0;
    return r;
  }

  static Field negate(const Field& x) {
    Field r;

    if(ff==NULL)
      ff=new FF();
    ((FF*)ff)->_negate(r.l, x.l);
    return r;
  }

  static Field add(const Field& a, const Field& b) {
    Field r;

    if(ff==NULL)
      ff=new FF();
    ((FF*)ff)->_add(r.l, a.l, b.l);
    return r;
  }

  static Field sub(const Field& a, const Field& b) {
    Field r;

    if(ff==NULL)
      ff=new FF();
    ((FF*)ff)->_sub(r.l, a.l, b.l);
    return r;
  }

  static Field sqr(const Field& x) {
    Field r;

    if(ff==NULL)
      ff=new FF();
    ((FF*)ff)->_mul(r.l, x.l, x.l);
    return r;
  }

  static Field mul(const Field& a, const Field& b) {
    Field r;

    if(ff==NULL)
      ff=new FF();
    ((FF*)ff)->_mul(r.l, a.l, b.l);
    return r;
  }

  static Field reduce(const Field& x) {
    Field r;

    // removes multiples of p from x

    if(ff==NULL)
      ff=new FF();
    ((FF*)ff)->_reduce(r.l, x.l);
    return r;
  }

  static Field reduce(const uint64_t* words, uint32_t wordCount) {
    Field r;

    // removes multiples of p from x

    if(ff==NULL)
      ff=new FF();
    ((FF*)ff)->_reduce(r.l, words, wordCount);
    return r;
  }

  static Field inv(const Field& x) {
    Field r;

    if(ff==NULL)
      ff=new FF();
    ((FF*)ff)->_inv(r.l, x.l);
    return r;
  }

  static Field toMontgomery(const Field& x) {
    Field r;

    if(ff==NULL)
      ff=new FF();
    ((FF*)ff)->_toMontgomery(r.l, x.l);
    return r;
  }

  static Field fromMontgomery(const Field& x) {
    Field r;

    if(ff==NULL)
      ff=new FF();
    ((FF*)ff)->_fromMontgomery(r.l, x.l);
    return r;
  }

  static Field fromString(const char* cString) {
    Field r;

    if(ff==NULL)
      ff=new FF();
    ((FF*)ff)->_fromString(r.l, cString);
    return r;
  }

  static Field random() {
    return Random::randomField<Field>();
  }
};

template<typename Field, bool montgomery>
void* FF<Field,montgomery>::ff=NULL;
