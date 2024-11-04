/***

Copyright (c) 2024, Ingonyama, Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#pragma once
#include <cstdint>

static constexpr uint32_t POLY_SIZE = 1 << 22;
static constexpr uint32_t EVAL_SIZE = 1 << 25;
static constexpr uint32_t EVAL_PLUS_SIZE = EVAL_SIZE + 8;
static constexpr uint32_t MERKLE_HEIGHT = 15;
static constexpr uint32_t MERKLE_NODE_COUNT = (1 << MERKLE_HEIGHT) - 1;
static constexpr uint32_t N_LEAF_NODES = 1 << (MERKLE_HEIGHT - 1);
static constexpr uint32_t N_NON_LEAF_NODES = MERKLE_NODE_COUNT - N_LEAF_NODES;
