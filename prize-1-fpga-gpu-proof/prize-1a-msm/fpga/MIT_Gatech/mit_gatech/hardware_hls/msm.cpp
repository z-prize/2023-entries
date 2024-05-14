#include"msm.hpp"
// #define VERBOSE_DEBUG
// #define VERBOSE_DEBUG_DBL

// #define VERBOSE_DEBUG // Print out detailed intermediate results for debugging purpose

///// BLS12-377 Curve
//! Curve information:
//! * Base ap_uint<NUM_OVERALL_BITWIDTH> : q = 258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177
//! * Scalar ap_uint<NUM_OVERALL_BITWIDTH> : r = 8444461749428370424248824938781546531375899335154063827935233455917409239041
//! * valuation(q - 1, 2) = 46
//! * valuation(r - 1, 2) = 47
//! * G1 curve equation: y^2 = x^3 + 1
//! * G2 curve equation: y^2 = x^3 + B, where
//!    * B = Fq2(0, 155198655607781456406391640216936120121836107652948796323930557600032281009004493664981332883744016074664192874906)

/////  BLS12-381 Curve
//! Curve information:
//! * Base ap_uint<NUM_OVERALL_BITWIDTH> : q = 4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787
//! * Scalar ap_uint<NUM_OVERALL_BITWIDTH> : r = 52435875175126190479447740508185965837690552500527637822603658699938581184513
//! * valuation(q - 1, 2) = 1
//! * valuation(r - 1, 2) = 32
//! * G1 curve equation: y^2 = x^3 + 4
//! * G2 curve equation: y^2 = x^3 + Fq2(4, 4)

// I follow the video here to implement the following code: https://www.google.com/search?sca_esv=d2e1279bafa3c31c&rlz=1C5CHFA_enUS1072US1072&tbm=vid&sxsrf=ACQVn0_PPMh9jbaEB8q86mJadJ7qYFmC7Q:1710426149951&q=weierstrass+elliptic+curve+scalar+multiplication&sa=X&ved=2ahUKEwjr5dP4-fOEAxXNGtAFHWJfDOEQ8ccDegQICxAI&biw=1708&bih=912&dpr=2#fpstate=ive&vld=cid:8a85a018,vid:iydGkrjJkSM,st:0
// I hope it works XDD


void ReadFromMem(
    ap_uint<NUM_OVERALL_BITWIDTH>               *x_array                             ,
    ap_uint<NUM_OVERALL_BITWIDTH>               *y_array                             ,
    ap_uint<NUM_OVERALL_BITWIDTH>               *scalar_array                        ,
    hls::stream<ap_uint<NUM_OVERALL_BITWIDTH> > x_array_stream[PARALLEL_DEGREE]      ,
    hls::stream<ap_uint<NUM_OVERALL_BITWIDTH> > y_array_stream[PARALLEL_DEGREE]      ,
    hls::stream<ap_uint<NUM_OVERALL_BITWIDTH> > scalar_array_stream[PARALLEL_DEGREE] ,
    int                                         degree                
){
  for (int i=0; i< degree; i=i+1){
    for (int j=0; j<PARALLEL_DEGREE; j=j+1){
      x_array_stream[j].write(x_array[i]);
      y_array_stream[j].write(y_array[i]);
      scalar_array_stream[j].write(scalar_array[i]);
#ifdef VERBOSE_DEBUG
       printf("%d-th input point: \n X: %Zx%Zx%Zx%Zx%Zx%Zx\n Y: %Zx%Zx%Zx%Zx%Zx%Zx\n --------------------\n", i, (uint64_t)x_array[i].range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)x_array[i].range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)x_array[i].range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)x_array[i].range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)x_array[i].range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)x_array[i].range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH), (uint64_t)y_array[i].range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)y_array[i].range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)y_array[i].range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)y_array[i].range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)y_array[i].range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)y_array[i].range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif
      // scalar_array_stream[j].write(scalar_array[i]);
    }
  }
};


ap_uint<NUM_OVERALL_BITWIDTH> modInverse(ap_uint<NUM_OVERALL_BITWIDTH> in_a, ap_uint<NUM_OVERALL_BITWIDTH> in_m){
    ap_uint<NUM_OVERALL_BITWIDTH> result=0;
    ap_int<NUM_OVERALL_BITWIDTH> a=in_a, m=in_m;
    ap_int<NUM_OVERALL_BITWIDTH> m0=in_m;
    ap_int<NUM_OVERALL_BITWIDTH> y=0, x=1, q=0, t1=0;
    ap_int<NUM_OVERALL_BITWIDTH> t2=0, a1=0, m1=0, y1=0, x1=0, q_times_y=0;

#ifdef VERBOSE_DEBUG
     printf("a %Zx%Zx%Zx%Zx%Zx%Zx%Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)a.range(12*BREAKDOWN_BITWIDTH-1, 11*BREAKDOWN_BITWIDTH), (uint64_t)a.range(11*BREAKDOWN_BITWIDTH-1, 10*BREAKDOWN_BITWIDTH), (uint64_t)a.range(10*BREAKDOWN_BITWIDTH-1, 9*BREAKDOWN_BITWIDTH), (uint64_t)a.range(9*BREAKDOWN_BITWIDTH-1, 8*BREAKDOWN_BITWIDTH), (uint64_t)a.range(8*BREAKDOWN_BITWIDTH-1, 7*BREAKDOWN_BITWIDTH), (uint64_t)a.range(7*BREAKDOWN_BITWIDTH-1, 6*BREAKDOWN_BITWIDTH), (uint64_t)a.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)a.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)a.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)a.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)a.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)a.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif
#ifdef VERBOSE_DEBUG
    printf("m %Zx%Zx%Zx%Zx%Zx%Zx%Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)m.range(12*BREAKDOWN_BITWIDTH-1, 11*BREAKDOWN_BITWIDTH), (uint64_t)m.range(11*BREAKDOWN_BITWIDTH-1, 10*BREAKDOWN_BITWIDTH), (uint64_t)m.range(10*BREAKDOWN_BITWIDTH-1, 9*BREAKDOWN_BITWIDTH), (uint64_t)m.range(9*BREAKDOWN_BITWIDTH-1, 8*BREAKDOWN_BITWIDTH), (uint64_t)m.range(8*BREAKDOWN_BITWIDTH-1, 7*BREAKDOWN_BITWIDTH), (uint64_t)m.range(7*BREAKDOWN_BITWIDTH-1, 6*BREAKDOWN_BITWIDTH), (uint64_t)m.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)m.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)m.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)m.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)m.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)m.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif
    if (m == 1)
      return 0;

    while (a > 1) {
        // q is quotient
        y1=0, x1=0, q_times_y=0;
        t2 = y;
#ifdef VERBOSE_DEBUG
         printf("a %Zx%Zx%Zx%Zx%Zx%Zx%Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)a.range(12*BREAKDOWN_BITWIDTH-1, 11*BREAKDOWN_BITWIDTH), (uint64_t)a.range(11*BREAKDOWN_BITWIDTH-1, 10*BREAKDOWN_BITWIDTH), (uint64_t)a.range(10*BREAKDOWN_BITWIDTH-1, 9*BREAKDOWN_BITWIDTH), (uint64_t)a.range(9*BREAKDOWN_BITWIDTH-1, 8*BREAKDOWN_BITWIDTH), (uint64_t)a.range(8*BREAKDOWN_BITWIDTH-1, 7*BREAKDOWN_BITWIDTH), (uint64_t)a.range(7*BREAKDOWN_BITWIDTH-1, 6*BREAKDOWN_BITWIDTH), (uint64_t)a.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)a.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)a.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)a.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)a.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)a.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif

        q =  a / m;
#ifdef VERBOSE_DEBUG
         printf("q %Zx%Zx%Zx%Zx%Zx%Zx%Zx%Zx%Zx%Zx%Zx%Zx\n",  (uint64_t)q.range(12*BREAKDOWN_BITWIDTH-1, 11*BREAKDOWN_BITWIDTH), (uint64_t)q.range(11*BREAKDOWN_BITWIDTH-1, 10*BREAKDOWN_BITWIDTH), (uint64_t)q.range(10*BREAKDOWN_BITWIDTH-1, 9*BREAKDOWN_BITWIDTH), (uint64_t)q.range(9*BREAKDOWN_BITWIDTH-1, 8*BREAKDOWN_BITWIDTH), (uint64_t)q.range(8*BREAKDOWN_BITWIDTH-1, 7*BREAKDOWN_BITWIDTH), (uint64_t)q.range(7*BREAKDOWN_BITWIDTH-1, 6*BREAKDOWN_BITWIDTH), (uint64_t)q.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)q.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)q.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)q.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)q.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)q.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif

        // m is remainder now
        m1 = a % m;
        a1 = m;
#ifdef VERBOSE_DEBUG
         printf("m1 %Zx%Zx%Zx%Zx%Zx%Zx%Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)m1.range(12*BREAKDOWN_BITWIDTH-1, 11*BREAKDOWN_BITWIDTH), (uint64_t)m1.range(11*BREAKDOWN_BITWIDTH-1, 10*BREAKDOWN_BITWIDTH), (uint64_t)m1.range(10*BREAKDOWN_BITWIDTH-1, 9*BREAKDOWN_BITWIDTH), (uint64_t)m1.range(9*BREAKDOWN_BITWIDTH-1, 8*BREAKDOWN_BITWIDTH), (uint64_t)m1.range(8*BREAKDOWN_BITWIDTH-1, 7*BREAKDOWN_BITWIDTH), (uint64_t)m1.range(7*BREAKDOWN_BITWIDTH-1, 6*BREAKDOWN_BITWIDTH), (uint64_t)m1.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)m1.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)m1.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)m1.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)m1.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)m1.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif

        // Update y and x
        q_times_y = q * y;
        y1 = x - q_times_y;
        x1 = t2;

        // Update final results
        m = m1;
        a = a1;
        x = x1;
        y = y1;
    }
#ifdef VERBOSE_DEBUG
     printf("finish the loop %Zx%Zx%Zx%Zx%Zx%Zx\n",  (uint64_t)x.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)x.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)x.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)x.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)x.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)x.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif
#ifdef VERBOSE_DEBUG
       printf("a %Zx%Zx%Zx%Zx%Zx%Zx\n",  (uint64_t)a.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)a.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)a.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)a.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)a.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)a.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif

    // Make x positive
    if (x < 0){
        result = x + m0;

#ifdef VERBOSE_DEBUG
         printf("enter the branch %Zx%Zx%Zx%Zx%Zx%Zx\n",  (uint64_t)result.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)result.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)result.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)result.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)result.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)result.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif
    }else{
      result = x;
    }

    return result;
}


void point_doubling(ap_uint<NUM_OVERALL_BITWIDTH> &rlt_x_q,
                    ap_uint<NUM_OVERALL_BITWIDTH> &rlt_y_q,
                    ap_uint<NUM_OVERALL_BITWIDTH> x,     
                    ap_uint<NUM_OVERALL_BITWIDTH> y){
  ap_uint<NUM_OVERALL_BITWIDTH>  q;
  q.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH) = Q_VALUE0;
  q.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH) = Q_VALUE1;
  q.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH) = Q_VALUE2;
  q.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH) = Q_VALUE3;
  q.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH) = Q_VALUE4;
  q.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH) = Q_VALUE5;
  q.range(NUM_OVERALL_BITWIDTH-1, BASE_BITWIDTH) = 0;
#ifdef VERBOSE_DEBUG
   printf("Modulus: %Zx%Zx%Zx%Zx%Zx%Zx\n  --------------------\n", (uint64_t)q.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)q.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)q.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)q.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)q.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)q.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif

  ap_uint<NUM_OVERALL_BITWIDTH>  scalar, result;
  ap_uint<NUM_OVERALL_BITWIDTH>  const_2, const_3, const_R; 
  ap_uint<NUM_OVERALL_BITWIDTH>  x_square, x_square_q, x_square_times_3, x_square_times_3_temp, x_square_times_3_q, y_times_2, y_times_2_q, beta;
  ap_uint<NUM_OVERALL_BITWIDTH>  beta_q, beta_square, beta_square_q, beta_square_q_sub_x, beta_square_q_sub_x_q;
  ap_uint<NUM_OVERALL_BITWIDTH>  rlt_x, rlt_y;//, rlt_x_q, rlt_y_q;
  ap_uint<NUM_OVERALL_BITWIDTH>  x_sub_rst_x, x_sub_rst_x_q, x_sub_rst_x_beta, x_sub_rst_x_beta_q, x_sub_rst_x_beta_q_sub_y, y_times_2_q_inverse_tmp, y_times_2_q_inverse;
  
#ifdef VERBOSE_DEBUG
   printf(" Input Point: \n X: %Zx%Zx%Zx%Zx%Zx%Zx%Zx\n Y: %Zx%Zx%Zx%Zx%Zx%Zx%Zx\n --------------------\n", (uint64_t)x.range(7*BREAKDOWN_BITWIDTH-1, 6*BREAKDOWN_BITWIDTH), (uint64_t)x.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)x.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)x.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)x.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)x.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)x.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH), (uint64_t)y.range(7*BREAKDOWN_BITWIDTH-1, 6*BREAKDOWN_BITWIDTH), (uint64_t)y.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)y.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)y.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)y.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)y.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)y.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif

  // (x^2) % q
  x_square =  x * x;
#ifdef VERBOSE_DEBUG_DBL
   printf(" x_square: %Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)x_square.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)x_square.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)x_square.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)x_square.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)x_square.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)x_square.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif

  x_square_q =  x_square % q;
#ifdef VERBOSE_DEBUG_DBL
   printf(" x_square_q: %Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)x_square_q.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)x_square_q.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)x_square_q.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)x_square_q.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)x_square_q.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)x_square_q.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif

  // (3 * x^2) % q
  x_square_times_3 = x_square_q * 3;
  if (x_square_times_3 > q) {x_square_times_3_temp = x_square_times_3 - q;}
  else {x_square_times_3_temp  = x_square_times_3;}
  if (x_square_times_3_temp > q) {x_square_times_3_q = x_square_times_3_temp - q;}
  else {x_square_times_3_q = x_square_times_3_temp;}
#ifdef VERBOSE_DEBUG_DBL
if (x_square_times_3_q > q) { 
   printf("error x_square_times_3_q > q");}
#endif

  // (2 * y) % q
  y_times_2 = (y << 1);
  if (y_times_2_q > q) { y_times_2_q = y_times_2 - q;}
  else {y_times_2_q = y_times_2;}
#ifdef VERBOSE_DEBUG_DBL
  if (y_times_2_q > q) { 
   printf("error y_times_2_q > q");}
#endif
#ifdef VERBOSE_DEBUG_DBL
   printf(" y_times_2_q: %Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)y_times_2_q.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)y_times_2_q.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)y_times_2_q.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)y_times_2_q.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)y_times_2_q.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)y_times_2_q.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif

  y_times_2_q_inverse = modInverse(y_times_2_q, q);

#ifdef VERBOSE_DEBUG_DBL
   printf(" y_times_2_q_inverse: %Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)y_times_2_q_inverse.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)y_times_2_q_inverse.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)y_times_2_q_inverse.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)y_times_2_q_inverse.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)y_times_2_q_inverse.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)y_times_2_q_inverse.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif

  // For curve y^2 =  x^3 + ax + b
  // Given point (x_1, y_1)
  // When tangent: beta = (3*x_1^2 + a)/2y_1
  beta = x_square_times_3_q * y_times_2_q_inverse;
  beta_q = beta % q;
#ifdef VERBOSE_DEBUG_DBL
   printf(" beta_q: %Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)beta_q.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)beta_q.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)beta_q.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)beta_q.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)beta_q.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)beta_q.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif

  beta_square = beta_q * beta_q;
  beta_square_q = beta_square % q;
#ifdef VERBOSE_DEBUG_DBL
   printf(" beta_square_q: %Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)beta_square_q.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)beta_square_q.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)beta_square_q.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)beta_square_q.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)beta_square_q.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)beta_square_q.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif
  
  //  @ToDo inversion is wrong.
  // mpz_mod_sub(&beta_square_q_sub_x_q, beta_square_q, x, q);
  if (beta_square_q > x){ beta_square_q_sub_x = beta_square_q - x;}
  else {beta_square_q_sub_x = beta_square_q - x + q;}
#ifdef VERBOSE_DEBUG_DBL
   printf(" beta_square_q_sub_x: %Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)beta_square_q_sub_x.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)beta_square_q_sub_x.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)beta_square_q_sub_x.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)beta_square_q_sub_x.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)beta_square_q_sub_x.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)beta_square_q_sub_x.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif

  beta_square_q_sub_x_q = beta_square_q_sub_x % q;
#ifdef VERBOSE_DEBUG
   printf(" beta_square_q_sub_x_q: %Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)beta_square_q_sub_x_q.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)beta_square_q_sub_x_q.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)beta_square_q_sub_x_q.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)beta_square_q_sub_x_q.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)beta_square_q_sub_x_q.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)beta_square_q_sub_x_q.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif

  // mpz_mod_sub(&rlt_x_q, beta_square_q_sub_x_q, x, q);
  if (beta_square_q_sub_x_q > x){ rlt_x = beta_square_q_sub_x_q - x;}
  else {rlt_x = beta_square_q_sub_x_q - x + q;}
  rlt_x_q = rlt_x % q;
#ifdef VERBOSE_DEBUG_DBL
   printf(" rlt_x_q: %Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)rlt_x_q.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)rlt_x_q.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)rlt_x_q.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)rlt_x_q.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)rlt_x_q.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)rlt_x_q.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif

  // mpz_mod_sub(&x_sub_rst_x_q, &x, rlt_x_q, q);
  if (x > rlt_x_q){x_sub_rst_x = x - rlt_x_q;}
  else {x_sub_rst_x = x - rlt_x_q + q;}
  x_sub_rst_x_q = x_sub_rst_x % q;
#ifdef VERBOSE_DEBUG_DBL
   printf(" x: %Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)x.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)x.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)x.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)x.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)x.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)x.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
   printf(" x_sub_rst_x_q: %Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)x_sub_rst_x_q.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)x_sub_rst_x_q.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)x_sub_rst_x_q.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)x_sub_rst_x_q.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)x_sub_rst_x_q.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)x_sub_rst_x_q.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif

  x_sub_rst_x_beta = x_sub_rst_x_q * beta_q;
  x_sub_rst_x_beta_q = x_sub_rst_x_beta % q;
#ifdef VERBOSE_DEBUG_DBL
   printf(" x_sub_rst_x_beta_q: %Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)x_sub_rst_x_beta_q.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)x_sub_rst_x_beta_q.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)x_sub_rst_x_beta_q.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)x_sub_rst_x_beta_q.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)x_sub_rst_x_beta_q.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)x_sub_rst_x_beta_q.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
   printf(" y: %Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)y.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)y.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)y.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)y.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)y.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)y.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif

  // mpz_mod_sub(&rlt_y_q, x_sub_rst_x_beta_q, y, q);
  if (x_sub_rst_x_beta_q > y){x_sub_rst_x_beta_q_sub_y = x_sub_rst_x_beta_q - y;}
  else {x_sub_rst_x_beta_q_sub_y = x_sub_rst_x_beta_q - y + q;}
  rlt_y_q = x_sub_rst_x_beta_q_sub_y % q;

#ifdef VERBOSE_DEBUG_DBL
   printf(" x_sub_rst_x_beta_q_sub_y: %Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)x_sub_rst_x_beta_q_sub_y.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)x_sub_rst_x_beta_q_sub_y.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)x_sub_rst_x_beta_q_sub_y.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)x_sub_rst_x_beta_q_sub_y.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)x_sub_rst_x_beta_q_sub_y.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)x_sub_rst_x_beta_q_sub_y.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
   printf(" rlt_y_q: %Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)rlt_y_q.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)rlt_y_q.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)rlt_y_q.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)rlt_y_q.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)rlt_y_q.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)rlt_y_q.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif

#ifdef VERBOSE_DEBUG_DBL
   printf(" Doubling Result: \n X: %Zx%Zx%Zx%Zx%Zx%Zx\n Y: %Zx%Zx%Zx%Zx%Zx%Zx\n --------------------\n", (uint64_t)rlt_x_q.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)rlt_x_q.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)rlt_x_q.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)rlt_x_q.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)rlt_x_q.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)rlt_x_q.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH), (uint64_t)rlt_y_q.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)rlt_y_q.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)rlt_y_q.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)rlt_y_q.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)rlt_y_q.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)rlt_y_q.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif

};

void point_addition(ap_uint<NUM_OVERALL_BITWIDTH> &x3,
                    ap_uint<NUM_OVERALL_BITWIDTH> &y3,
                    ap_uint<NUM_OVERALL_BITWIDTH> x1,
                    ap_uint<NUM_OVERALL_BITWIDTH> y1,
                    ap_uint<NUM_OVERALL_BITWIDTH> x2,
                    ap_uint<NUM_OVERALL_BITWIDTH> y2
  ){
  ap_uint<NUM_OVERALL_BITWIDTH>  q;
  q.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH) = Q_VALUE0;
  q.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH) = Q_VALUE1;
  q.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH) = Q_VALUE2;
  q.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH) = Q_VALUE3;
  q.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH) = Q_VALUE4;
  q.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH) = Q_VALUE5;
  q.range(NUM_OVERALL_BITWIDTH-1, BASE_BITWIDTH) = 0;
  // beta = (y2 - y1) / (x2 - x1)
  // x3 = beta^2 - x1 - x2
  // y3 = beta * (x1 - x3) - y1
#ifdef VERBOSE_DEBUG
   printf(" x1: %Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)x1.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)x1.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)x1.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)x1.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)x1.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)x1.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif
#ifdef VERBOSE_DEBUG
   printf(" y1: %Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)y1.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)y1.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)y1.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)y1.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)y1.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)y1.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif
#ifdef VERBOSE_DEBUG
   printf(" x2: %Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)x2.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)x2.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)x2.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)x2.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)x2.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)x2.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif
#ifdef VERBOSE_DEBUG
   printf(" y2: %Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)y2.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)y2.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)y2.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)y2.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)y2.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)y2.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif

  ap_uint<NUM_OVERALL_BITWIDTH>  y2_y1=0, x2_x1=0, y2_y1_q=0, x2_x1_q=0, x2_x1_q_inv=0, x2_x1_q_inv_q=0;
  ap_uint<NUM_OVERALL_BITWIDTH>  beta=0, beta_q=0, beta_q_square=0, beta_q_square_q=0;
  ap_uint<NUM_OVERALL_BITWIDTH>  beta_q_square_q_x1=0, beta_q_square_q_x1_q=0, beta_q_square_q_x1_q_x2=0, beta_q_square_q_x1_q_x2_q=0; // (x3=beta_q_square_q_x1_q_x2_q)
  ap_uint<NUM_OVERALL_BITWIDTH>  x1_x3=0, x1_x3_q=0, beta_q_x1_x3_q=0, beta_q_x1_x3_q_q=0, beta_q_x1_x3_q_q_y1=0, beta_q_x1_x3_q_q_y1_q=0; // (y3=beta_x1_x3_q_q_y1)

  // beta = (y2 - y1) / (x2 - x1)
  if (y2 > y1){ y2_y1 = y2 - y1;}
  else {y2_y1 = y2 - y1 + q;}
  y2_y1_q = y2_y1 % q;
#ifdef VERBOSE_DEBUG
   printf(" y2-y1: %Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)y2_y1_q.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)y2_y1_q.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)y2_y1_q.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)y2_y1_q.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)y2_y1_q.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)y2_y1_q.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif

  if (x2 > x1){ x2_x1 = x2 - x1;}
  else {x2_x1 = x2 - x1 + q;}
  x2_x1_q = x2_x1 % q;
#ifdef VERBOSE_DEBUG
   printf(" x2-x1: %Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)x2_x1_q.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)x2_x1_q.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)x2_x1_q.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)x2_x1_q.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)x2_x1_q.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)x2_x1_q.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif

  x2_x1_q_inv = modInverse(x2_x1_q, q);
#ifdef VERBOSE_DEBUG
   printf(" x2_x1_q_inv: %Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)x2_x1_q_inv.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)x2_x1_q_inv.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)x2_x1_q_inv.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)x2_x1_q_inv.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)x2_x1_q_inv.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)x2_x1_q_inv.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif

  // mpz_invert(x2_x1_q_inv, x2_x1_q, q);
  x2_x1_q_inv_q = x2_x1_q_inv % q;
#ifdef VERBOSE_DEBUG
   printf(" x2_x1_q_inv_q: %Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)x2_x1_q_inv_q.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)x2_x1_q_inv_q.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)x2_x1_q_inv_q.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)x2_x1_q_inv_q.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)x2_x1_q_inv_q.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)x2_x1_q_inv_q.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif
  
  beta = x2_x1_q_inv_q * y2_y1_q;
  beta_q = beta % q;
#ifdef VERBOSE_DEBUG
   printf(" beta_q: %Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)beta_q.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)beta_q.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)beta_q.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)beta_q.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)beta_q.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)beta_q.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif

  // x3 = beta^2 - x1 - x2
  beta_q_square = beta_q * beta_q;
  beta_q_square_q = beta_q_square % q;

  if (beta_q_square_q > x1){beta_q_square_q_x1 = beta_q_square_q - x1;}
  else {beta_q_square_q_x1 = beta_q_square_q - x1 + q;}
  beta_q_square_q_x1_q = beta_q_square_q_x1 % q;
#ifdef VERBOSE_DEBUG
   printf(" beta_q_square_q_x1_q: %Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)beta_q_square_q_x1_q.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)beta_q_square_q_x1_q.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)beta_q_square_q_x1_q.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)beta_q_square_q_x1_q.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)beta_q_square_q_x1_q.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)beta_q_square_q_x1_q.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif

  if (beta_q_square_q_x1_q > x2){beta_q_square_q_x1_q_x2 = beta_q_square_q_x1_q - x2;}
  else {beta_q_square_q_x1_q_x2 = beta_q_square_q_x1_q - x2 + q;}
  x3 = beta_q_square_q_x1_q_x2 % q;

  // y3 = beta * (x1 - x3) - y1
  if (x1 > x3){x1_x3 = x1 - x3;}
  else {x1_x3 = x1 - x3 + q;}
  x1_x3_q = x1_x3 % q;

  beta_q_x1_x3_q = beta_q * x1_x3_q;
  beta_q_x1_x3_q_q = beta_q_x1_x3_q % q;
#ifdef VERBOSE_DEBUG
   printf(" beta_q_x1_x3_q_q: %Zx%Zx%Zx%Zx%Zx%Zx\n", (uint64_t)beta_q_x1_x3_q_q.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)beta_q_x1_x3_q_q.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)beta_q_x1_x3_q_q.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)beta_q_x1_x3_q_q.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)beta_q_x1_x3_q_q.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)beta_q_x1_x3_q_q.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif

  if (beta_q_x1_x3_q_q > y1){beta_q_x1_x3_q_q_y1 = beta_q_x1_x3_q_q - y1;}
  else {beta_q_x1_x3_q_q_y1 = beta_q_x1_x3_q_q - y1 + q;}
  y3 = beta_q_x1_x3_q_q_y1 % q;

#ifdef VERBOSE_DEBUG
   printf(" Accumulated Result: \n X: %Zx%Zx%Zx%Zx%Zx%Zx\n Y: %Zx%Zx%Zx%Zx%Zx%Zx\n --------------------\n", (uint64_t)x3.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)x3.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)x3.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)x3.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)x3.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)x3.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH), (uint64_t)y3.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)y3.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)y3.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)y3.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)y3.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)y3.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
#endif

#ifdef VERBOSE_DEBUG
   printf(" Addition Result: \n X: %Zx\n Y: %Zx\n --------------------\n", x3, y3);
#endif
}


void  scalar_multiplication(hls::stream<ap_uint<NUM_OVERALL_BITWIDTH> > acc_x_array_stream[PARALLEL_DEGREE],
                            hls::stream<ap_uint<NUM_OVERALL_BITWIDTH> > acc_y_array_stream[PARALLEL_DEGREE],
                            hls::stream<ap_uint<NUM_OVERALL_BITWIDTH> > x_array_stream[PARALLEL_DEGREE],
                            hls::stream<ap_uint<NUM_OVERALL_BITWIDTH> > y_array_stream[PARALLEL_DEGREE],
                            hls::stream<ap_uint<NUM_OVERALL_BITWIDTH> > scalar_array_stream[PARALLEL_DEGREE],
                            int                                         degree
){
  ap_uint<NUM_OVERALL_BITWIDTH> x;// = x_array_stream[0].read();
  ap_uint<NUM_OVERALL_BITWIDTH> y;// = y_array_stream[0].read();
  ap_uint<NUM_OVERALL_BITWIDTH> scalar;// = scalar_array_stream[0].read();
  // ap_uint<NUM_OVERALL_BITWIDTH> scalar = scalar_array_stream[0].read();
  ap_uint<NUM_OVERALL_BITWIDTH> dbl_x=0, dbl_y=0;
  ap_uint<NUM_OVERALL_BITWIDTH> acc_x=0, acc_y=0;
  ap_uint<NUM_OVERALL_BITWIDTH> cur_acc_x=0, cur_acc_y=0;
  ap_uint<NUM_OVERALL_BITWIDTH> cur_base_x=x, cur_base_y=y;
    
  for(int degree_idx = 0; degree_idx < degree; degree_idx = degree_idx + 1){
    x = x_array_stream[degree_idx].read();
    y = y_array_stream[degree_idx].read();
    scalar = scalar_array_stream[degree_idx].read();

    for (int i = 0; i < SCALAR_ITERATION_BIT-1; ++i) {  
      // printf("%d-th bit ", i);
      if(scalar[i]==1){
          // printf("SET!!\n");

        if(acc_x == 0 && acc_y == 0){
          acc_x = cur_base_x; acc_y = cur_base_y;
        }
        else{
          cur_acc_x = acc_x;
          cur_acc_y = acc_y;
          point_addition(acc_x, acc_y, cur_acc_x, cur_acc_y, dbl_x, dbl_y);
        }
      }
      if(i < SCALAR_ITERATION_BIT - 1){
        point_doubling(dbl_x, dbl_y, cur_base_x, cur_base_y);
        cur_base_x = dbl_x;
        cur_base_y = dbl_y;
      }
    }
    printf(" ScalarMultiplication Result (WB): \n X: %Zx%Zx%Zx%Zx%Zx%Zx\n Y: %Zx%Zx%Zx%Zx%Zx%Zx\n --------------------\n", (uint64_t)acc_x.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)acc_x.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)acc_x.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)acc_x.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)acc_x.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)acc_x.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH), (uint64_t)acc_y.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)acc_y.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)acc_y.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)acc_y.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)acc_y.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)acc_y.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));

    acc_x_array_stream[0].write(acc_x);
    acc_y_array_stream[0].write(acc_y);
  }
};


void  WriteBack(hls::stream<ap_uint<NUM_OVERALL_BITWIDTH> > acc_x_array_stream[PARALLEL_DEGREE],
                hls::stream<ap_uint<NUM_OVERALL_BITWIDTH> > acc_y_array_stream[PARALLEL_DEGREE],
                int                                         degree
){
  ap_uint<NUM_OVERALL_BITWIDTH> acc_x, acc_y, final_x=0, final_y=0, cur_x=0, cur_y=0;
  bool initial_addition = true;
  for (int i=0; i < degree; i++){
    acc_x = acc_x_array_stream[0].read();
    acc_y = acc_y_array_stream[0].read();
    if (initial_addition){
      final_x = acc_x;
      final_y = acc_y;
      initial_addition = false;
    }else{
      cur_x = final_x;
      cur_y = final_y;

      point_addition(final_x, final_y, cur_x, cur_y, acc_x, acc_y);
    }

    printf(" Final Result (WB): \n X: %Zx%Zx%Zx%Zx%Zx%Zx\n Y: %Zx%Zx%Zx%Zx%Zx%Zx\n --------------------\n", (uint64_t)final_x.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)final_x.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)final_x.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)final_x.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)final_x.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)final_x.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH), (uint64_t)final_y.range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)final_y.range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)final_y.range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)final_y.range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)final_y.range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)final_y.range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));
  }
}

void msm(
  ap_uint<NUM_OVERALL_BITWIDTH>                                          *x_array                                 ,
  ap_uint<NUM_OVERALL_BITWIDTH>                                          *y_array                                 ,
  ap_uint<NUM_OVERALL_BITWIDTH>                                          *scalar_array                            ,
  int                                                                    degree
){
  #pragma HLS INTERFACE m_axi     depth=MAX_DEGREE   offset=slave  port=x_array       bundle=x_array
  #pragma HLS INTERFACE m_axi     depth=MAX_DEGREE   offset=slave  port=y_array       bundle=y_array
  #pragma HLS INTERFACE m_axi     depth=MAX_DEGREE   offset=slave  port=scalar_array  bundle=scalar_array
  #pragma HLS INTERFACE s_axilite            port=degree
  #pragma HLS INTERFACE s_axilite  register  port=return

  hls::stream<ap_uint<NUM_OVERALL_BITWIDTH>  >                          x_array_stream[PARALLEL_DEGREE]           ;
  #pragma HLS STREAM variable=x_array_stream depth=HLS_STREAM_DEPTH type=fifo
  hls::stream<ap_uint<NUM_OVERALL_BITWIDTH>  >                          y_array_stream[PARALLEL_DEGREE]           ;
  #pragma HLS STREAM variable=y_array_stream depth=HLS_STREAM_DEPTH type=fifo

  hls::stream<ap_uint<NUM_OVERALL_BITWIDTH>  >                          scalar_array_stream[PARALLEL_DEGREE]      ;
  #pragma HLS STREAM variable=scalar_array_stream depth=HLS_STREAM_DEPTH type=fifo

  ReadFromMem(x_array, y_array, scalar_array, x_array_stream, y_array_stream,  scalar_array_stream, degree);

  hls::stream<ap_uint<NUM_OVERALL_BITWIDTH>  >                          acc_x_array_stream[PARALLEL_DEGREE]  ;
  #pragma HLS STREAM variable=doubling_x_array_stream depth=HLS_STREAM_DEPTH type=fifo
  hls::stream<ap_uint<NUM_OVERALL_BITWIDTH>  >                          acc_y_array_stream[PARALLEL_DEGREE]  ;
  #pragma HLS STREAM variable=doubling_y_array_stream depth=HLS_STREAM_DEPTH type=fifo

  scalar_multiplication(acc_x_array_stream, acc_y_array_stream, x_array_stream, y_array_stream, scalar_array_stream, degree);

  WriteBack(acc_x_array_stream, acc_y_array_stream, degree);

  // hls::stream<ap_uint<NUM_OVERALL_BITWIDTH>  >                          doubling_x_array_stream[PARALLEL_DEGREE]  ;
  // #pragma HLS STREAM variable=doubling_x_array_stream depth=HLS_STREAM_DEPTH type=fifo
  // hls::stream<ap_uint<NUM_OVERALL_BITWIDTH>  >                          doubling_y_array_stream[PARALLEL_DEGREE]  ;
  // #pragma HLS STREAM variable=doubling_y_array_stream depth=HLS_STREAM_DEPTH type=fifo

  // point_doubling(doubling_x_array_stream, doubling_y_array_stream, x_array_stream, y_array_stream);


  // hls::stream<ap_uint<NUM_OVERALL_BITWIDTH>  >                          acc_x_array_stream[PARALLEL_DEGREE]  ;
  // #pragma HLS STREAM variable=doubling_x_array_stream depth=HLS_STREAM_DEPTH type=fifo
  // hls::stream<ap_uint<NUM_OVERALL_BITWIDTH>  >                          acc_y_array_stream[PARALLEL_DEGREE]  ;
  // #pragma HLS STREAM variable=doubling_y_array_stream depth=HLS_STREAM_DEPTH type=fifo

  // point_addition(acc_x_array_stream, acc_y_array_stream, doubling_x_array_stream, doubling_y_array_stream, x_cur_point_stream, y_cur_point_stream);
  
  // Test point doubling
}
