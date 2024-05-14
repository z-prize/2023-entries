#pragma once
#include "hw_msm.h"

struct test_t {
    int valid;
    int sub;
    int ec_bucket_inf;
    int ec_point_inf;
    int out_inf;
    field_t ec_bucket_x;
    field_t ec_bucket_y;
    field_t ec_point_x;
    field_t ec_point_y;
    field_t ec_untree;
    field_t ec_inverse;
    field_t dx;
    field_t dy;
    field_t dx_inv;
    field_t lambda;
    field_t out_x;
    field_t out_y;
};

static test_t test[130][336] = {
#include "test_file.cpp"
};
