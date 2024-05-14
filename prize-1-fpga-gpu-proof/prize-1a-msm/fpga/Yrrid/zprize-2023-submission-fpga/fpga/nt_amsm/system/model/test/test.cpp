/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <omp.h>
#include "hw_msm.h"
#include "k_main.h"
#include "k_unified_proc.h"
#include "k_inverter.h"
#include "HostFields.h"
#include "HostCurves.h"


#ifndef __BLS12_377__
using host_point_xy_t = Host::Curve<Host::BLS12381G1>::PointXY;
using host_point_xyzz_t = Host::Curve<Host::BLS12381G1>::PointXYZZ;
static constexpr char TEST_SCALARS_PATH[] = "../../../data/scalars_381.hex";
static constexpr char TEST_POINTS_PATH[] = "../../../data/points_381.hex";
#else
using host_point_xy_t = Host::Curve<Host::BLS12377G1>::PointXY;
using host_point_xyzz_t = Host::Curve<Host::BLS12377G1>::PointXYZZ;
static constexpr char TEST_SCALARS_PATH[] = "../../../data/scalars_377.hex";
static constexpr char TEST_POINTS_PATH[] = "../../../data/points_377.hex";
#endif

extern "C" {
struct device_point_t {
    uint64_t l[12];
};

struct device_scalar_t {
    uint64_t l[4];
};
}

host_point_xyzz_t window_sum(const device_point_t window[4096]) {
    host_point_xyzz_t sum, sum_of_sums;
    host_point_xy_t point;
    for (int i = 4095; i >= 0; i--) {
        point.load(window[i].l);
        point.fromDeviceMontgomeryToHostMontgomery();
        sum.addXY(point);
        sum_of_sums.addXYZZ(sum);
    }
    return sum_of_sums;
}

host_point_xy_t final_sum(const host_point_xyzz_t window_sums[21]) {
    host_point_xyzz_t sum;
    for (int w = 21 - 1; w >= 0; w--) {
        for (int j = 0; j < 13; j++) {
            sum.dbl();
        }
        sum.addXYZZ(window_sums[w]);
    }
    return sum.normalize();
}

std::vector<scalar_t> read_scalars(const std::string &filename) {
    std::vector<scalar_t> result;

    std::ifstream file(filename);

    std::string word;

    while (file >> word) {
        result.emplace_back(scalar_t(("0x" + word).c_str(), 16));
    }

    file.close();
    return result;
}

std::vector<point_t> read_points(const std::string &filename) {
    std::vector<point_t> result;

    std::ifstream file(filename);

    std::string word1, word2;

    while (file >> word1 >> word2) {
        point_t tmp;
        tmp.x = field_t(("0x" + word1).c_str(), 16);
        tmp.y = field_t(("0x" + word2).c_str(), 16);
        result.emplace_back(tmp);
    }

    file.close();
    return result;
}

template<bls12_t CURVE>
void model(
    const ap_uint<256> *scalars,
    const ap_uint<256> *points,
    ap_uint<256> *res,
    int msm_size,
    int full_batches
) {
    hls_thread_local hls::stream<main_to_proc_flit_t> main_to_proc_0("main_to_proc_0");
    hls_thread_local hls::stream<main_to_proc_flit_t> main_to_proc_1("main_to_proc_1");
    hls_thread_local hls::stream<main_to_proc_flit_t> main_to_proc_2("main_to_proc_2");
    hls_thread_local hls::stream<proc_status_t> proc_0_to_main_status("proc_0_to_main_status");
    hls_thread_local hls::stream<proc_status_t> proc_1_to_main_status("proc_1_to_main_status");
    hls_thread_local hls::stream<proc_status_t> proc_2_to_main_status("proc_2_to_main_status");
    hls_thread_local hls::stream<proc_to_main_flit_t> proc_0_to_main_data("proc_0_to_main_data");
    hls_thread_local hls::stream<proc_to_main_flit_t> proc_1_to_main_data("proc_1_to_main_data");
    hls_thread_local hls::stream<proc_to_main_flit_t> proc_2_to_main_data("proc_2_to_main_data");

    hls_thread_local hls::stream<field_t> proc_0_to_inverter("proc_0_to_inverter");
    hls_thread_local hls::stream<field_t> proc_1_to_inverter("proc_1_to_inverter");
    hls_thread_local hls::stream<field_t> proc_2_to_inverter("proc_2_to_inverter");

    hls_thread_local hls::stream<field_t> inverter_to_proc_0("inverter_to_proc_0");
    hls_thread_local hls::stream<field_t> inverter_to_proc_1("inverter_to_proc_1");
    hls_thread_local hls::stream<field_t> inverter_to_proc_2("inverter_to_proc_2");

    k_inverter<CURVE>::k_inverter_inst(
        proc_0_to_inverter,
        proc_1_to_inverter,
        proc_2_to_inverter,
        inverter_to_proc_0,
        inverter_to_proc_1,
        inverter_to_proc_2
    );
    if constexpr (CURVE == bls12_381) {
        k_bls12_381_unified_proc_0(main_to_proc_0, inverter_to_proc_0, proc_0_to_main_status, proc_0_to_main_data,
                                   proc_0_to_inverter);
        k_bls12_381_unified_proc_1(main_to_proc_1, inverter_to_proc_1, proc_1_to_main_status, proc_1_to_main_data,
                                   proc_1_to_inverter);
        k_bls12_381_unified_proc_2(main_to_proc_2, inverter_to_proc_2, proc_2_to_main_status, proc_2_to_main_data,
                                   proc_2_to_inverter);
    } else if (CURVE == bls12_377) {
        k_bls12_377_unified_proc_0(main_to_proc_0, inverter_to_proc_0, proc_0_to_main_status, proc_0_to_main_data,
                                   proc_0_to_inverter);
        k_bls12_377_unified_proc_1(main_to_proc_1, inverter_to_proc_1, proc_1_to_main_status, proc_1_to_main_data,
                                   proc_1_to_inverter);
        k_bls12_377_unified_proc_2(main_to_proc_2, inverter_to_proc_2, proc_2_to_main_status, proc_2_to_main_data,
                                   proc_2_to_inverter);
    }
    k_main(
        scalars,
        points,
        res,
        msm_size,
        full_batches,
        proc_0_to_main_status,
        proc_1_to_main_status,
        proc_2_to_main_status,
        proc_0_to_main_data,
        proc_1_to_main_data,
        proc_2_to_main_data,
        main_to_proc_0,
        main_to_proc_1,
        main_to_proc_2
    );
    spdlog::info("Flushing SIM...");
    usleep(1000000);
}

ap_uint<768> to_raw(const point_t &point) {
    ap_uint<768> tmp;
    tmp(383, 0) = point.x;
    tmp(767, 384) = point.y;
    return tmp;
}

point_t from_raw(const ap_uint<768> &raw) {
    point_t tmp;
    tmp.x = raw(383, 0);
    tmp.y = raw(767, 384);
    return tmp;
}

int main() {
    constexpr int MSM_SIZE = 8192;
    constexpr int FULL_BATCHES = MSM_SIZE / PRE_SLICE_BATCH_SIZE;
    omp_set_num_threads(8);
#pragma omp parallel
    {
    }
    // -- [prepare scalars] --------------------------------------------------------------------------------------------
    auto test_scalars = new device_scalar_t[MSM_SIZE];
    std::ifstream scalars_file(TEST_SCALARS_PATH);

    for (int i = 0; i < MSM_SIZE; i++) {
        std::string line;
        getline(scalars_file, line);
        Host::readHex(test_scalars[i].l, 4, line.c_str());
    }
    scalars_file.close();
    std::cout << "Loaded scalars.hex" << std::endl;
    // -----------------------------------------------------------------------------------------------------------------
    // -- [prepare points] ---------------------------------------------------------------------------------------------
    auto test_points = new device_point_t[MSM_SIZE];
    std::ifstream points_file(TEST_POINTS_PATH);

    for (int i = 0; i < MSM_SIZE; i++) {
        std::string line;
        getline(points_file, line);
        host_point_xy_t tmp = host_point_xy_t::fromCString(line.c_str());
        tmp.store(test_points[i].l);
    }
    points_file.close();
    std::cout << "Loaded points.hex" << std::endl;
    // -----------------------------------------------------------------------------------------------------------------

    auto buckets = new device_point_t[21][4096];
    auto start = std::chrono::high_resolution_clock::now();
#ifndef __BLS12_377__
    model<bls12_381>(reinterpret_cast<ap_uint<256> *>(test_scalars), reinterpret_cast<ap_uint<256> *>(test_points),
                     reinterpret_cast<ap_uint<256> *>(buckets), MSM_SIZE, FULL_BATCHES);
#else
    model<bls12_377>(reinterpret_cast<ap_uint<256> *>(test_scalars), reinterpret_cast<ap_uint<256> *>(test_points),
                reinterpret_cast<ap_uint<256> *>(buckets), MSM_SIZE, FULL_BATCHES);
#endif
    auto end = std::chrono::high_resolution_clock::now();


    host_point_xyzz_t window_sums[21];

    // auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (int i = 0; i < 21; i++) {
        window_sums[i] = window_sum(buckets[i]);
    }
    host_point_xy_t msm_res = final_sum(window_sums);
    msm_res.fromHostMontgomeryToDeviceMontgomery();
    msm_res.dump(false);

    // Correct bls12_381
    // X = 0CE1F9073B0818E85811A527B07F59B0678A52E76CE3D31162BA483C8C5386EF731C7B7DE5960041124AF03D6ADD1A61
    // Y = 076E6D530890455A74271085F16803EF6E32C9764CF3DA6C16FCB0FBD66A12B77EC611386E3DF6C7B9F0111FFF242F0B

    // Correct bls12_377
    // X = 9906D449A558C906B1DE1F9920657AC845BB66461127D0A5B0319A6F9F29C9940C1EB9FFA53035DCB59DCA24FE67B4
    // Y = 127B37FDB09A19694B6760E40E360C35EF3617CB87A531C82464D3FD6766DDC1B58D6A5820CB8584EE73A25B6E81DAB

    std::cout << "Done" << std::endl;
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time: " << duration.count() << " us" << std::endl;

    return 0;
}
