/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#pragma once

void k_bls12_381_unified_proc_0(
    hls::stream<main_to_proc_flit_t> &from_main,
    hls::stream<field_t> &from_inverter,
    hls::stream<proc_status_t> &to_main_status,
    hls::stream<proc_to_main_flit_t> &to_main_data,
    hls::stream<field_t> &to_inverter
);

void k_bls12_381_unified_proc_1(
    hls::stream<main_to_proc_flit_t> &from_main,
    hls::stream<field_t> &from_inverter,
    hls::stream<proc_status_t> &to_main_status,
    hls::stream<proc_to_main_flit_t> &to_main_data,
    hls::stream<field_t> &to_inverter
);

void k_bls12_381_unified_proc_2(
    hls::stream<main_to_proc_flit_t> &from_main,
    hls::stream<field_t> &from_inverter,
    hls::stream<proc_status_t> &to_main_status,
    hls::stream<proc_to_main_flit_t> &to_main_data,
    hls::stream<field_t> &to_inverter
);

void k_bls12_377_unified_proc_0(
    hls::stream<main_to_proc_flit_t> &from_main,
    hls::stream<field_t> &from_inverter,
    hls::stream<proc_status_t> &to_main_status,
    hls::stream<proc_to_main_flit_t> &to_main_data,
    hls::stream<field_t> &to_inverter
);

void k_bls12_377_unified_proc_1(
    hls::stream<main_to_proc_flit_t> &from_main,
    hls::stream<field_t> &from_inverter,
    hls::stream<proc_status_t> &to_main_status,
    hls::stream<proc_to_main_flit_t> &to_main_data,
    hls::stream<field_t> &to_inverter
);

void k_bls12_377_unified_proc_2(
    hls::stream<main_to_proc_flit_t> &from_main,
    hls::stream<field_t> &from_inverter,
    hls::stream<proc_status_t> &to_main_status,
    hls::stream<proc_to_main_flit_t> &to_main_data,
    hls::stream<field_t> &to_inverter
);
