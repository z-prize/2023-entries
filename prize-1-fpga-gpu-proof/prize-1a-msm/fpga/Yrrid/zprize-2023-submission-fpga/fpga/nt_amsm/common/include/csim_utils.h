/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#pragma once

#ifndef __SYNTHESIS__

#include "spdlog/spdlog.h"

#define CSIM_INFO(X) spdlog::info("{}_{}_{}_{}: {}", KERNEL_NAME, KERNEL_ID, COMPONENT_NAME, COMPONENT_ID, X)
#define CSIM_ERROR(X) spdlog::error("{}_{}_{}_{}: {}", KERNEL_NAME, KERNEL_ID, COMPONENT_NAME, COMPONENT_ID, X)
#define CSIM_INFO_ARGS(FMT, ...) spdlog::info("{}_{}_{}_{}: {}", KERNEL_NAME, KERNEL_ID, COMPONENT_NAME, COMPONENT_ID, fmt::format(FMT, __VA_ARGS__))
#define CSIM_USLEEP(X) usleep(X)
#define CSIM_ERROR_NO_LABEL(X) spdlog::error(X)

#else
#define CSIM_INFO(X)
#define CSIM_ERROR(X)
#define CSIM_INFO_ARGS(FMT, ...)
#define CSIM_USLEEP(X)
#define CSIM_ERROR_NO_LABEL(X)
#endif
