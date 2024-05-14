template<typename Field, bool montgomery>
class FF;

template<typename Curve, bool montgomery>
class PointXY;

class Random {
   public:
   static void *random;

   gmp_randstate_t rstate;
   mpz_t           zr;

   Random() {
     gmp_randinit_mt(rstate);
     mpz_init(zr);
   }

   ~Random() {
      gmp_randclear(rstate);
      mpz_clear(zr);
   }

   void _setSeed(uint64_t seed) {
     gmp_randseed_ui(rstate, seed);
   }

   void _randomBits(uint64_t* r, uint32_t bits) {
      uint32_t count;

      mpz_urandomb(zr, rstate, bits);
      for(int i=0;i<(bits+63)/64;i++)
        r[i]=0;
      mpz_export(r, NULL, -1, 8, 0, 0, zr);
   }

   static void setSeed(uint64_t seed) {
     if(random==NULL)
       random=new Random();
     ((Random*)random)->_setSeed(seed);
   }

   static void randomBits(uint64_t* r, uint32_t bits) {
     if(random==NULL)
       random=new Random();
     ((Random*)random)->_randomBits(r, bits);
   }

   template<typename Field>
   static Field randomField() {
     uint64_t words[(Field::bits+63+32)/64];

     if(random==NULL)
       random=new Random();
     ((Random*)random)->_randomBits(words, Field::bits+32);
     return FF<Field,false>::reduce(words, (Field::bits+63+32)/64);
   }

   template<typename Curve,bool montgomery>
   static PointXY<Curve,montgomery> randomPoint() {
     typename Curve::OrderField scalar=randomField<typename Curve::OrderField>();

     return PointXY<Curve,montgomery>::generator().scale(scalar);
   }
};

void* Random::random=NULL;

