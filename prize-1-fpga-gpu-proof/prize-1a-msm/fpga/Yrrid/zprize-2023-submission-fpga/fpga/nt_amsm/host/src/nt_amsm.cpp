/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#include <cassert>
#include <vector>
#include <iostream>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include "HostFields.h"
#include "HostCurves.h"

static constexpr char BLS12_381_XCLBIN_PATH[] = "./nt_amsm_bls12_381.xclbin";
static constexpr char BLS12_377_XCLBIN_PATH[] = "./nt_amsm_bls12_377.xclbin";

struct device_point_t {
    uint64_t l[12];
};

struct device_scalar_t {
    uint64_t l[4];
};

enum curve_t {
    BLS12_381,
    BLS12_377
};

struct context_t {
    curve_t curve{};
    uint32_t max_batch_count{};
    uint32_t max_point_count{};
    xrt::device dev;
    xrt::kernel krnl;
    xrt::bo points_buffer;
    std::vector<xrt::bo> scalar_buffers;
    std::vector<xrt::bo> res_buffers;
};

extern "C" {
void *createContext(uint32_t curve, uint32_t maxBatchCount, uint32_t maxPointCount);

int32_t destroyContext(void *contextPtr);

int32_t preprocessPoints(void *contextPtr, void *pointData, uint32_t pointCount);

int32_t processBatches(void *contextPtr, void *resultData, void *scalarData, uint32_t batchCount, uint32_t pointCount);
}

using host_point_xy_t = Host::Curve<Host::BLS12381G1>::PointXY;
using host_point_xyzz_t = Host::Curve<Host::BLS12381G1>::PointXYZZ;


static host_point_xyzz_t window_sum(const device_point_t window[4096]) {
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

static host_point_xy_t final_sum(const host_point_xyzz_t window_sums[21]) {
    host_point_xyzz_t sum;
    for (int w = 21 - 1; w >= 0; w--) {
        for (int j = 0; j < 13; j++) {
            sum.dbl();
        }
        sum.addXYZZ(window_sums[w]);
    }
    return sum.normalize();
}

static host_point_xy_t post_processing(const device_point_t windows[21][4096]) {
    host_point_xyzz_t window_sums[21];
    for (int i = 0; i < 21; i++) {
        window_sums[i] = window_sum(windows[i]);
    }
    return final_sum(window_sums);
}

void *createContext(uint32_t curve, uint32_t maxBatchCount, uint32_t maxPointCount) {
    assert(curve == 377 || curve == 381);
    assert(maxBatchCount <= 8);
    assert(maxPointCount <= 67108864);

    auto *context = new context_t;
    context->curve = curve == 381 ? BLS12_381 : BLS12_377;
    context->max_batch_count = maxBatchCount;
    context->max_point_count = maxPointCount;

    const size_t scalar_buffer_size = maxPointCount * sizeof(device_scalar_t);
    const size_t points_buffer_size = maxPointCount * sizeof(device_point_t);
    constexpr size_t res_buffer_size = 4096 * 21 * sizeof(device_point_t);

    // load xclbin
    context->dev = xrt::device(0);
    if (context->curve == BLS12_381) {
        const auto xclbin_uuid = context->dev.load_xclbin(BLS12_381_XCLBIN_PATH);
        context->krnl = xrt::kernel(context->dev, xclbin_uuid, "k_main", xrt::kernel::cu_access_mode::exclusive);
    } else if (context->curve == BLS12_377) {
        const auto xclbin_uuid = context->dev.load_xclbin(BLS12_377_XCLBIN_PATH);
        context->krnl = xrt::kernel(context->dev, xclbin_uuid, "k_main", xrt::kernel::cu_access_mode::exclusive);
    }

    // allocate points buffer
    context->points_buffer = xrt::bo(context->dev, points_buffer_size, xrt::bo::flags::device_only,
                                     context->krnl.group_id(1));

    // allocate scalar and result buffers
    for (int i = 0; i < context->max_batch_count; i++) {
        auto scalars_buffer = xrt::bo(context->dev, scalar_buffer_size, xrt::bo::flags::device_only,
                                      context->krnl.group_id(0));
        context->scalar_buffers.push_back(scalars_buffer);
        auto res_buffer = xrt::bo(context->dev, res_buffer_size, xrt::bo::flags::normal, context->krnl.group_id(2));
        context->res_buffers.push_back(res_buffer);
    }

    return context;
}

int32_t destroyContext(void *contextPtr) {
    delete static_cast<context_t *>(contextPtr);
    return 0;
}

int32_t preprocessPoints(void *contextPtr, void *pointData, uint32_t pointCount) {
    auto context = static_cast<context_t *>(contextPtr);
    assert(pointCount < context->max_point_count);
    // copy points to device
    const size_t n_point_bytes = pointCount * sizeof(device_point_t);
    context->points_buffer.write(pointData, n_point_bytes, 0);
    context->points_buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return 0;
}



static int32_t optimized_batch_4(context_t *context, void *res_bytes, void *scalar_bytes, uint32_t msm_size,
                          uint32_t padding) {
    const size_t n_scalar_bytes = msm_size * sizeof(device_scalar_t);
    auto res = static_cast<host_point_xy_t *>(res_bytes);

    // copy first scalars to device and kick off 1st run
    context->scalar_buffers[0].write(scalar_bytes, n_scalar_bytes, sizeof(device_scalar_t) * 0 * msm_size);
    context->scalar_buffers[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run = context->krnl(context->scalar_buffers[0], context->points_buffer, context->res_buffers[0], msm_size,
                             padding);

    // copy the rest of the buffers
    context->scalar_buffers[1].write(scalar_bytes, n_scalar_bytes, sizeof(device_scalar_t) * 1 * msm_size);
    context->scalar_buffers[1].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    context->scalar_buffers[2].write(scalar_bytes, n_scalar_bytes, sizeof(device_scalar_t) * 2 * msm_size);
    context->scalar_buffers[2].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    context->scalar_buffers[3].write(scalar_bytes, n_scalar_bytes, sizeof(device_scalar_t) * 3 * msm_size);
    context->scalar_buffers[3].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    // wait for 1st run
    run.wait();
    // kick off 2nd run
    run = context->krnl(context->scalar_buffers[1], context->points_buffer, context->res_buffers[1], msm_size,
                             padding);
    // meanwhile compute final sum of 1st run
    auto windows = reinterpret_cast<device_point_t(*)[4096]>(context->res_buffers[0].map<device_point_t *>());
    res[0] = post_processing(windows);
    // wait for 2nd run
    run.wait();
    // kick off 3rd run
    run = context->krnl(context->scalar_buffers[2], context->points_buffer, context->res_buffers[2], msm_size,
                             padding);
    // meanwhile compute final sum of 2nd run
    windows = reinterpret_cast<device_point_t(*)[4096]>(context->res_buffers[1].map<device_point_t *>());
    res[1] = post_processing(windows);
    // wait for 3rd run
    run.wait();
    // kick off 4th run
    run = context->krnl(context->scalar_buffers[3], context->points_buffer, context->res_buffers[3], msm_size,
                                 padding);
    // meanwhile compute final sum of second run
    windows = reinterpret_cast<device_point_t(*)[4096]>(context->res_buffers[2].map<device_point_t *>());
    res[2] = post_processing(windows);
    // wait for 4rd run
    run.wait();
    // compute final sum of fourth run
    windows = reinterpret_cast<device_point_t(*)[4096]>(context->res_buffers[3].map<device_point_t *>());
    res[3] = post_processing(windows);
    return 0;
}


int32_t processBatches(void *contextPtr, void *resultData, void *scalarData, uint32_t batchCount, uint32_t pointCount) {
    // time critical code
    auto context = static_cast<context_t *>(contextPtr);

    static constexpr int PRE_SLICE_BATCH_SIZE = 48;

    const uint32_t msm_size = pointCount;
    const uint32_t padding = PRE_SLICE_BATCH_SIZE - msm_size % PRE_SLICE_BATCH_SIZE;

    if (batchCount == 4) {
        return optimized_batch_4(context, resultData, scalarData, msm_size, padding);
    }

    const size_t n_scalar_bytes = msm_size * sizeof(device_scalar_t);
    auto res = static_cast<host_point_xy_t *>(resultData);

    // TODO: Optimize batched processing
    for (int i = 0; i < batchCount; i++) {
        context->scalar_buffers[i].write(scalarData, n_scalar_bytes, sizeof(device_scalar_t) * i * msm_size);
        context->scalar_buffers[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto run = context->krnl(context->scalar_buffers[i], context->points_buffer, context->res_buffers[i], msm_size,
                                 padding);
        run.wait();
        const auto windows = reinterpret_cast<device_point_t(*)[4096]>(context->res_buffers[i].map<device_point_t *>());
        res[i] = post_processing(windows);
    }
    return 0;
}
