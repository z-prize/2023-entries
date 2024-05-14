#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define __gmp_const const

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"

// #define BLS12_377
#define BLS12_381 // Initially define BLS12_381, commented out for illustration

#ifdef BLS12_377
    #undef BLS12_381
#endif

#ifdef BLS12_381
    #undef BLS12_377
#endif



// Workload Parameters
constexpr int MAX_DEGREE                       = (1 << 24);

// Hardware Parameters
constexpr int PARALLEL_DEGREE                  = 1; 
constexpr int HLS_STREAM_DEPTH                 = 4; 
constexpr int BREAKDOWN_BITWIDTH               = 64; 


#ifdef BLS12_381
constexpr int NUM_OVERALL_BITWIDTH             = (384 << 1);
constexpr int BASE_BITWIDTH                    = 384;
constexpr int SCALAR_ITERATION_BIT             = 256;
// constexpr ap_uint<NUM_OVERALL_BITWIDTH>  Q_VALUE                        = 0x1A0111EA397FE69A4B1BA7B6434BACD764774B84F38512BF6730D2A0F6B0F6241EABFFFEB153FFFFB9FEFFFFFFFFAAAB;
constexpr uint64_t Q_VALUE0 = 0xB9FEFFFFFFFFAAAB;
constexpr uint64_t Q_VALUE1 = 0x1EABFFFEB153FFFF;
constexpr uint64_t Q_VALUE2 = 0x6730D2A0F6B0F624;
constexpr uint64_t Q_VALUE3 = 0x64774B84F38512BF;
constexpr uint64_t Q_VALUE4 = 0x4B1BA7B6434BACD7;
constexpr uint64_t Q_VALUE5 = 0x1A0111EA397FE69A;

#elif defined BLS12_377
constexpr int NUM_OVERALL_BITWIDTH             = (377 << 1);
constexpr int BASE_BITWIDTH                    = 377;
constexpr int SCALAR_ITERATION_BIT             = 256;
// constexpr ap_uint<NUM_OVERALL_BITWIDTH>  Q_VALUE                        = 0x1AE3A4617C510EAC63B05C06CA1493B1A22D9F300F5138F1EF3622FBA094800170B5D44300000008508C00000000001;
constexpr uint64_t Q_VALUE0 = 0x8508C00000000001;
constexpr uint64_t Q_VALUE1 = 0x170B5D4430000000;
constexpr uint64_t Q_VALUE2 = 0x1EF3622FBA094800;
constexpr uint64_t Q_VALUE3 = 0x1A22D9F300F5138F;
constexpr uint64_t Q_VALUE4 = 0xC63B05C06CA1493B;
constexpr uint64_t Q_VALUE5 = 0x1AE3A4617C510EA;
 
#endif

constexpr int HOST_DATA_ARRAY_LENGTH           = 32;

void ReadFromMem(
    ap_uint<NUM_OVERALL_BITWIDTH>               *x_array                             ,
    ap_uint<NUM_OVERALL_BITWIDTH>               *y_array                             ,
    ap_uint<NUM_OVERALL_BITWIDTH>               *scalar_array                        ,
    hls::stream<ap_uint<NUM_OVERALL_BITWIDTH> > x_array_stream[PARALLEL_DEGREE]      ,     
    hls::stream<ap_uint<NUM_OVERALL_BITWIDTH> > y_array_stream[PARALLEL_DEGREE]      ,                 
    hls::stream<ap_uint<NUM_OVERALL_BITWIDTH> > scalar_array_stream[PARALLEL_DEGREE] ,
    int                                         degree                
);

// void multi_scalar_multiplication(const char* points_csv_path, const char* scalars_csv_path);
void msm(
  ap_uint<NUM_OVERALL_BITWIDTH>                                          x_array[HOST_DATA_ARRAY_LENGTH]          ,
  ap_uint<NUM_OVERALL_BITWIDTH>                                          y_array[HOST_DATA_ARRAY_LENGTH]          ,
  ap_uint<NUM_OVERALL_BITWIDTH>                                          scalar_array[HOST_DATA_ARRAY_LENGTH]     ,
  int                                           degree
);