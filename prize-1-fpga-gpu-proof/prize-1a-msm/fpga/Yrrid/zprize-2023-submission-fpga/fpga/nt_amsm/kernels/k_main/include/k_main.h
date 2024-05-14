/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#pragma once

#include "hw_msm.h"

void k_main(
        const ap_uint<256> *scalars_mem,
        const ap_uint<256> *points_mem,
        ap_uint<256> *res_mem,
        int msm_size,
        int full_batches,
        hls::stream<proc_status_t> &from_proc_0_status,
        hls::stream<proc_status_t> &from_proc_1_flushed,
        hls::stream<proc_status_t> &from_proc_2_flushed,
        hls::stream<proc_to_main_flit_t> &from_proc_0_data,
        hls::stream<proc_to_main_flit_t> &from_proc_1_data,
        hls::stream<proc_to_main_flit_t> &from_proc_2_data,
        hls::stream<main_to_proc_flit_t> &to_proc_0,
        hls::stream<main_to_proc_flit_t> &to_proc_1,
        hls::stream<main_to_proc_flit_t> &to_proc_2
);
