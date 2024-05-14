/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <unistd.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include "HostFields.h"
#include "HostCurves.h"

#ifndef __BLS12_377__
using host_point_xy_t = Host::Curve<Host::BLS12381G1>::PointXY;
using host_point_xyzz_t = Host::Curve<Host::BLS12381G1>::PointXYZZ;
static constexpr char TEST_SCALARS_PATH[] = "../../data/scalars_381.hex";
static constexpr char TEST_POINTS_PATH[] = "../../data/points_381.hex";
static constexpr char XCLBIN_PATH[] = "../../system/hw/work/bls12_381/nt_amsm_bls12_381.xclbin";
#else
using host_point_xy_t = Host::Curve<Host::BLS12377G1>::PointXY;
using host_point_xyzz_t = Host::Curve<Host::BLS12377G1>::PointXYZZ;
static constexpr char TEST_SCALARS_PATH[] = "../../data/scalars_377.hex";
static constexpr char TEST_POINTS_PATH[] = "../../data/points_377.hex";
static constexpr char XCLBIN_PATH[] = "../../system/hw/work/bls12_377/nt_amsm_bls12_377.xclbin";
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

static constexpr int PRE_SLICE_BATCH_SIZE = 48;
static constexpr int MSM_SIZE = 8192;
static constexpr int SCALARS_BUFFER_SIZE = MSM_SIZE * sizeof(device_scalar_t);
static constexpr int POINTS_BUFFER_SIZE = MSM_SIZE * sizeof(device_point_t);
static constexpr int RES_BUFFER_SIZE = 21 * 4096 * sizeof(device_point_t);
constexpr int FULL_BATCHES = MSM_SIZE/PRE_SLICE_BATCH_SIZE;

int main() {
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

    // -- [XRT HOST CODE] ----------------------------------------------------------------------------------------------
    unsigned int dev_index = 0;
    auto device = xrt::device(dev_index);
    std::cout << "device name:     " << device.get_info<xrt::info::device::name>() << std::endl;
    std::cout << "device bdf:      " << device.get_info<xrt::info::device::bdf>() << std::endl;

    std::cout << "loading xclbin..." << std::endl;
    auto xclbin_uuid = device.load_xclbin(XCLBIN_PATH);
    std::cout << "xclbin uuid:     " << xclbin_uuid.to_string() << std::endl;
    auto krnl = xrt::kernel(device, xclbin_uuid, "k_main", xrt::kernel::cu_access_mode::exclusive);
    auto scalars_buffer = xrt::bo(device, SCALARS_BUFFER_SIZE, krnl.group_id(0));
    auto points_buffer = xrt::bo(device, POINTS_BUFFER_SIZE, krnl.group_id(1));
    auto res_buffer = xrt::bo(device, RES_BUFFER_SIZE, krnl.group_id(2));

    auto res = new device_point_t[21][4096];
    for (int i = 0; i < 21; i++) {
        for (int j = 0; j < 4096; j++) {
            for (int k = 0; k < 12; k++) {
                res[i][j].l[k] = 0;
            }
        }
    }
    res_buffer.write(res);
    res_buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    points_buffer.write(test_points);
    points_buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    scalars_buffer.write(test_scalars);
    scalars_buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "starting" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    auto run = krnl(scalars_buffer, points_buffer, res_buffer, MSM_SIZE, FULL_BATCHES);
    if (ERT_CMD_STATE_COMPLETED != run.wait(1000)) {
        std::cerr << "error: timeout..." << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();

    res_buffer.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    res_buffer.read(res);

    host_point_xyzz_t window_sums[21];
    for (int i = 0; i < 21; i++) {
        window_sums[i] = window_sum(res[i]);
    }
    host_point_xy_t msm_res = final_sum(window_sums);

    std::cout << "normal space msm result: " << std::endl;
    msm_res.dump();
    msm_res.fromHostMontgomeryToDeviceMontgomery();
    std::cout << "for comparison with MSM.cpp tool: " << std::endl;
    msm_res.dump(false);
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "device Processing Time: " << duration.count() << " us" << std::endl;
    // // -----------------------------------------------------------------------------------------------------------------
    return 0;
}
