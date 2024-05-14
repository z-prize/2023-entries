#include "msm.hpp"
#include <gmp.h>

/*
  ./program "/home/ubuntu/work/ZPrize-23-Prize1/Prize 1A/test_code/zprize_msm_curve_377_bases_dim_16_seed_0.csv" "/home/ubuntu/work/ZPrize-23-Prize1/Prize 1A/test_code/zprize_msm_curve_377_scalars_dim_16_seed_0.csv"
*/



/*
void scalar_multiplication(hls::stream<ap_uint<NUM_OVERALL_BITWIDTH> > result_point_x, hls::stream<ap_uint<NUM_OVERALL_BITWIDTH> > result_point_y, hls::stream<ap_uint<NUM_OVERALL_BITWIDTH> > base_point_x, hls::stream<ap_uint<NUM_OVERALL_BITWIDTH> > base_point_y, hls::stream<ap_uint<NUM_OVERALL_BITWIDTH> > scalar, hls::stream<ap_uint<NUM_OVERALL_BITWIDTH> > modulus_q) {

    // Temporary variables for intermediate results
    ap_uint<NUM_OVERALL_BITWIDTH>  current_point_x, current_point_y, temp_point_x, temp_point_y;
    ap_uint<NUM_OVERALL_BITWIDTH>  temp_acc_point_x, temp_acc_point_y;
  
    // Copy base point to current_point
    mpz_set(current_point_x, base_point_x);
    mpz_set(current_point_y, base_point_y);

    size_t num_bits = mpz_sizeinbase(scalar, 2);

    for (size_t i = 0; i < num_bits; ++i) {
        if (ap_uint<NUM_OVERALL_BITWIDTH> stbit(scalar, i)) {
            if (mpz_cmp_ui(*result_point_x, 0) == 0 && mpz_cmp_ui(*result_point_y, 0) == 0) {
                // result_point is the identity element, so just copy current_point to it
                mpz_set(*result_point_x, current_point_x);
                mpz_set(*result_point_y, current_point_y);
            } else {
                // Perform point addition
                ap_uint<NUM_OVERALL_BITWIDTH>  tmp_current_point_x, tmp_current_point_y;
                mpz_inits(tmp_current_point_x, tmp_current_point_y, NULL);
                mpz_set(tmp_current_point_x, *result_point_x);
                mpz_set(tmp_current_point_y, *result_point_y);
                point_addition(&temp_acc_point_x, &temp_acc_point_y,
                               tmp_current_point_x, tmp_current_point_y,
                               current_point_x, current_point_y, modulus_q);
                mpz_set(*result_point_x, temp_acc_point_x);
                mpz_set(*result_point_y, temp_acc_point_y);
            }
        }

        // Prepare for next iteration by doubling current_point
        if(i < num_bits - 1){
          mpz_set(temp_point_x, current_point_x);
          mpz_set(temp_point_y, current_point_y);
          point_doubling(&current_point_x, &current_point_y, temp_point_x, temp_point_y);
        }

# ifdefVERBOSE_DEBUG
#ifdef gmp_VERBOSE_DEBUG
       printf(" Current Result: \n X: %Zx\n Y: %Zx\n --------------------\n\n\n", result_point_x, result_point_y);
#endif
#endif    
    }

    // Clean up
    mpz_clear(current_point_x);
    mpz_clear(current_point_y);
    mpz_clear(temp_point_x);
    mpz_clear(temp_point_y);
}


// void parse_two_mpz_from_line(hls::stream<ap_uint<NUM_OVERALL_BITWIDTH> > out, hls::stream<ap_uint<NUM_OVERALL_BITWIDTH> > out2, const char* buffer) {
void parse_mpz_from_line(ap_uint<NUM_OVERALL_BITWIDTH> * out, ap_uint<NUM_OVERALL_BITWIDTH> * out2, const char* buffer) {
    char* value_start = strchr(buffer, '(');
    if (!value_start) return;

    value_start++; // Move past the quote
    char* value_end = strchr(value_start, ')');
    if (!value_end) return;

    size_t value_length = value_end - value_start;
    char* value_str = malloc(value_length + 1);
    strncpy(value_str, value_start, value_length);
    value_str[value_length] = '\0';
    mpz_set_str(out, value_str, 16);
    // Second part

    char* value_start2 = strchr(value_end, '(');
    value_start2++; // Move past the quote

    char* value_end2 = strchr(value_start2, ')');

    size_t value_length2 = value_end2 - value_start2;
    char* value_str2 = malloc(value_length2 + 1);
    strncpy(value_str2, value_start2, value_length2);
    value_str2[value_length2] = '\0';
    mpz_set_str(out2, value_str2, 16);
#ifdef VERBOSE_DEBUG
#ifdef gmp_VERBOSE_DEBUG
     printf(" input X: %Zx \n Y: %Zx \n", out, out2);
#endif
#endif
    free(value_str2);
}

// Assumes buffer is a line from your CSV containing an ap_uint<NUM_OVERALL_BITWIDTH>  in the format zz: FpXXX "value"
void parse_mpz_from_line(ap_uint<NUM_OVERALL_BITWIDTH> * out, const char* buffer) {
// void parse_mpz_from_line(hls::stream<ap_uint<NUM_OVERALL_BITWIDTH> > out, const char* buffer) {
    char* value_start = strchr(buffer, '(');
    if (!value_start) return;

    value_start++; // Move past the quote
    char* value_end = strchr(value_start, ')');
    if (!value_end) return;
    buffer = value_end;

    size_t value_length = value_end - value_start;
    char* value_str = malloc(value_length + 1);
    strncpy(value_str, value_start, value_length);
    value_str[value_length] = '\0';
    mpz_set_str(out, value_str, 16);

#ifdef VERBOSE_DEBUG
     printf("scalar string: %s \n", value_str);
#endif
#ifdef gmp_VERBOSE_DEBUG
     printf(" scalar: %Zx \n", out);
#endif

    free(value_str);
}


void multi_scalar_multiplication(const char* points_csv_path, const char* scalars_csv_path) {
    FILE* points_file = fopen(points_csv_path, "r");
    FILE* scalars_file = fopen(scalars_csv_path, "r");
    if (!points_file || !scalars_file) {
        // Handle error
        return;
    }

    char points_buffer[2048];
    char scalars_buffer[1024];
    mpz_t result_x, result_y, copy_result_x, copy_result_y, q;
    mpz_inits(result_x, result_y, copy_result_x, copy_result_y, NULL);
  #ifdef BLS12_381
    mpz_init_set_str(q, "4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787", 10); // BLS12-381
  #elif defined BLS12_377
    mpz_init_set_str(q, "258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177", 10); // BLS12-377
  #endif
    bool initial_addition = true;
    while (fgets(points_buffer, sizeof(points_buffer), points_file) && fgets(scalars_buffer, sizeof(scalars_buffer), scalars_file)) {
        mpz_t point_x, point_y, scalar, cur_result_x, cur_result_y;
        mpz_inits(point_x, point_y, scalar, cur_result_x, cur_result_y,  NULL);

        // Assume modulus_q is set correctly for your curve
        // mpz_set_str(modulus_q, "Your modulus here", 10);

        parse_two_mpz_from_line(&point_x, &point_y, points_buffer);
        parse_mpz_from_line(&scalar, scalars_buffer);
#ifdef gmp_VERBOSE_DEBUG
         printf(" Input X: %Zx\n Y: %Zx \n* scalar: %Zx\n --------------------\n", point_x, point_y, scalar);
#endif

        // Perform scalar multiplication
        scalar_multiplication(&cur_result_x, &cur_result_y, point_x, point_y, scalar, q);
#ifdef gmp_VERBOSE_DEBUG
         printf(" cur_result:\n   X: %Zx\n   Y: %Zx\n --------------------\n", cur_result_x, cur_result_y);
#endif

        if(initial_addition){
          mpz_set(result_x, cur_result_x);
          mpz_set(result_y, cur_result_y);
          initial_addition = false;
        }
        else{
          mpz_set(copy_result_x, result_x);
          mpz_set(copy_result_y, result_y);
          point_addition(&result_x, &result_y, cur_result_x, cur_result_y, copy_result_x, copy_result_y, q);
        }
        // Print or store result_x and result_y as needed

        mpz_clears(point_x, point_y, scalar, cur_result_x, cur_result_y, NULL);
    }
#ifdef gmp_VERBOSE_DEBUG
     printf(" Final Result:\n   X: %Zx\n   Y: %Zx\n --------------------\n", result_x, result_y);
#endif

    fclose(points_file);
    fclose(scalars_file);
}


int main_bk_read_from_CLI(int argc, char *argv[]) {
  if (argc != 3) {
#ifdef VERBOSE_DEBUG
       printf("Usage: %s <input_x> <input_y>\n", argv[0]);
#endif
      return 1; // Exit with an error code
  }

  // Assign command line arguments to input_x and input_y
  char* point_input_file = argv[1];
  char* scalar_input_file = argv[2];

  multi_scalar_multiplication(point_input_file, scalar_input_file);

  return 0;
}
*/


/*
int main(int argc, char *argv[]) {
  if (argc != 3) {
      printf("Usage: %s <input_x> <input_y>\n", argv[0]);
      return 1; // Exit with an error code
  }

  // Assign command line arguments to input_x and input_y
  char* point_input_file = argv[1];
  char* scalar_input_file = argv[2];

  multi_scalar_multiplication(point_input_file, scalar_input_file);

  return 0;
}


void multi_scalar_multiplication(const char* points_csv_path, const char* scalars_csv_path) {
    FILE* points_file = fopen(points_csv_path, "r");
    FILE* scalars_file = fopen(scalars_csv_path, "r");
    if (!points_file || !scalars_file) {
        // Handle error
        return;
    }


    char points_buffer[2048];
    char scalars_buffer[1024];
    mpz_t result_x, result_y, copy_result_x, copy_result_y, q;
    mpz_inits(result_x, result_y, copy_result_x, copy_result_y, NULL);
  #ifdef BLS12_381
    mpz_init_set_str(q, "4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787", 10); // BLS12-381
  #elif defined BLS12_377
    mpz_init_set_str(q, "258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177", 10); // BLS12-377
  #endif
    bool initial_addition = true;
    while (fgets(points_buffer, sizeof(points_buffer), points_file) && fgets(scalars_buffer, sizeof(scalars_buffer), scalars_file)) {
        mpz_t point_x, point_y, scalar, cur_result_x, cur_result_y;
        mpz_inits(point_x, point_y, scalar, cur_result_x, cur_result_y,  NULL);

        // Assume modulus_q is set correctly for your curve
        // mpz_set_str(modulus_q, "Your modulus here", 10);

        parse_two_mpz_from_line(&point_x, &point_y, points_buffer);
        parse_mpz_from_line(&scalar, scalars_buffer);
#ifdef VERBOSE_DEBUG
        gmp_printf(" Input X: %Zx\n Y: %Zx \n* scalar: %Zx\n --------------------\n", point_x, point_y, scalar);
#endif

        // Perform scalar multiplication
        scalar_multiplication(&cur_result_x, &cur_result_y, point_x, point_y, scalar, q);
#ifdef VERBOSE_DEBUG
        gmp_printf(" cur_result:\n   X: %Zx\n   Y: %Zx\n --------------------\n", cur_result_x, cur_result_y);
#endif

        if(initial_addition){
          mpz_set(result_x, cur_result_x);
          mpz_set(result_y, cur_result_y);
          initial_addition = false;
        }
        else{
          mpz_set(copy_result_x, result_x);
          mpz_set(copy_result_y, result_y);
          point_addition(&result_x, &result_y, cur_result_x, cur_result_y, copy_result_x, copy_result_y, q);
        }
        // Print or store result_x and result_y as needed

        mpz_clears(point_x, point_y, scalar, cur_result_x, cur_result_y, NULL);
    }
    gmp_printf(" Final Result:\n   X: %Zx\n   Y: %Zx\n --------------------\n", result_x, result_y);

    fclose(points_file);
    fclose(scalars_file);
}


void specific_value_test(){
  char* input_x = "16A86B7BADC3C532B1ACE6D225E0A80425DDF81EE696F1C03FE66BDD58F1F1A787E5FE407C838FAD95721F5E3B8AB1D2";
  char* input_y = "00E835EB1E998D7F12AA207B0312815073EAF5C31879BD08F0CB2F8B48D32012EDF384B4B18AAA536B45C453C8AF46D8";
  
  int a = 0, b = 4;

  mpz_t x, y;
  mpz_t scalar, q;

#ifdef BLS12_381
  mpz_init_set_str(q, "4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787", 10); // BLS12-381
#elif defined BLS12_377
  mpz_init_set_str(q, "258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177", 10); // BLS12-377
#endif
  mpz_init_set_str(x, input_x, 16);
  mpz_init_set_str(y, input_y, 16);
  mpz_init_set_str(scalar, "0000000000000000000000000000000000000000000000000000000000000005", 16);

#ifdef VERBOSE_DEBUG
  gmp_printf(" Input \n X: %Zx\n Y: %Zx \n * scalar: %Zx\n --------------------\n", x, y, scalar);
#endif

  mpz_t result_x_q, result_y_q;
  scalar_multiplication(&result_x_q, &result_y_q, x, y, scalar, q);

#ifdef VERBOSE_DEBUG
  gmp_printf(" Final Result:\n X: %Zx\n Y: %Zx\n --------------------\n", result_x_q, result_y_q);
#endif
}
*/


int main() {
  ap_uint<NUM_OVERALL_BITWIDTH> x_array[HOST_DATA_ARRAY_LENGTH], y_array[HOST_DATA_ARRAY_LENGTH], scalar_array[HOST_DATA_ARRAY_LENGTH]; 
  // x_array[0] = 16A86B7BADC3C532B1ACE6D225E0A80425DDF81EE696F1C03FE66BDD58F1F1A787E5FE407C838FAD95721F5E3B8AB1D2
  for (int i =0; i<HOST_DATA_ARRAY_LENGTH; i=i+1){
    x_array[0] = 0; 
    y_array[0] = 0; 
    scalar_array[i] = 0; 
  }
  x_array[0].range(NUM_OVERALL_BITWIDTH-1, BASE_BITWIDTH) = 0;
  x_array[0].range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH) = 0x12B8F3ABF50782B1;//0xF8A94D761852712;
  x_array[0].range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH) = 0x8F37410B10CF408E;//0xCC9408E3B2802AAD;
  x_array[0].range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH) = 0x88B7749A40E344F5;//0xFAC6AE8840E33DC0;
  x_array[0].range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH) = 0x62F7CC171612DAA1;//0xB02C3DF6BF3C139B;
  x_array[0].range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH) = 0x981B9BEAE6981802;//0xD9390F10BD7E1942;
  x_array[0].range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH) = 0x2993BCDEB42AF53;//0xD0A4EE1E2BCE3C4C;

  x_array[1].range(NUM_OVERALL_BITWIDTH-1, BASE_BITWIDTH) = 0;
  x_array[1].range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH) = 0x4C1A1C869D16404;
  x_array[1].range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH) = 0x4F09F9A42A10E448;
  x_array[1].range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH) = 0x8A99ADF06A5A689F;
  x_array[1].range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH) = 0xABFD76890A137A88;
  x_array[1].range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH) = 0x4ADF415D51661575;
  x_array[1].range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH) = 0x8B2CB3FB68E8E601;

  // y_array[0] = 00E835EB1E998D7F12AA207B0312815073EAF5C31879BD08F0CB2F8B48D32012EDF384B4B18AAA536B45C453C8AF46D8;
  uint64_t temp = 0x6B45C453C8AF46D8;
  printf("last chunk of temp: %Zx \n", temp);

  temp = 0xEDF384B4B18AAA53;
  y_array[0].range(NUM_OVERALL_BITWIDTH-1, BASE_BITWIDTH) = 0;
  y_array[0].range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH) = 0x15800FA0BA4AEFB8; //0x7BDBF9FC4F33DA6;
  y_array[0].range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH) = 0xAF1A7CA4AF195117; //0xF1B22B6F60B0B6F4;
  y_array[0].range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH) = 0x99FB01492444A070; //0x32B8CB2B90FA74F5;
  y_array[0].range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH) = 0xD485C7A3FE9B22BC; //0x5994DAC2421EB2C3;
  y_array[0].range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH) = 0xFABB6BC2007F76A3; //0xA967FFDC83400E2F;
  y_array[0].range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH) = 0xADC6560ECF990A47; //0xF2A8F92996AB87C0;

  y_array[1].range(NUM_OVERALL_BITWIDTH-1, BASE_BITWIDTH) = 0;
  y_array[1].range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH) = 0x9846E9776D3EEAC;
  y_array[1].range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH) = 0xE43F1B26A71CFFC0;
  y_array[1].range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH) = 0xF84D021168AC96BB;
  y_array[1].range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH) = 0xF32B0037DAD49449;
  y_array[1].range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH) = 0xA3259DF6DC4A9542;
  y_array[1].range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH) = 0xDAEC9D18D6AD2078;

  scalar_array[0].range(NUM_OVERALL_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH) = 0;
  scalar_array[0].range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH) = 0x57AA5DF37B9BD97A;
  scalar_array[0].range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH) = 0x5E5F84F4797EAC33;
  scalar_array[0].range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH) = 0xE5EBE0C6E2CA2FBC;
  scalar_array[0].range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH) = 0xA1B3B3D7052CE35D;

  scalar_array[1].range(NUM_OVERALL_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH) = 0;
  scalar_array[1].range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH) = 0x43131D0617D95A6F;
  scalar_array[1].range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH) = 0xBD46C1F9055F60E8;
  scalar_array[1].range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH) = 0x28ACAAE2E6E7E50;
  scalar_array[1].range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH) = 0xA471ED47553ECFE;

  printf("In TB: 0-th input point: \n X: %Zx%Zx%Zx%Zx%Zx%Zx\n Y: %Zx%Zx%Zx%Zx%Zx%Zx\n --------------------\n", (uint64_t)x_array[0].range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)x_array[0].range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)x_array[0].range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)x_array[0].range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)x_array[0].range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)x_array[0].range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH), (uint64_t)y_array[0].range(6*BREAKDOWN_BITWIDTH-1, 5*BREAKDOWN_BITWIDTH), (uint64_t)y_array[0].range(5*BREAKDOWN_BITWIDTH-1, 4*BREAKDOWN_BITWIDTH), (uint64_t)y_array[0].range(4*BREAKDOWN_BITWIDTH-1, 3*BREAKDOWN_BITWIDTH), (uint64_t)y_array[0].range(3*BREAKDOWN_BITWIDTH-1, 2*BREAKDOWN_BITWIDTH), (uint64_t)y_array[0].range(2*BREAKDOWN_BITWIDTH-1, 1*BREAKDOWN_BITWIDTH), (uint64_t)y_array[0].range(1*BREAKDOWN_BITWIDTH-1, 0*BREAKDOWN_BITWIDTH));

  int degree = 1;
  msm(x_array, y_array, scalar_array, degree);

  return 0;
}

