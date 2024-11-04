//////////////////////////////////////////////////////////////////////////////////
// (C) Copyright Ponos Technology 2024
// All Rights Reserved
// *** Ponos Technology Confidential ***
// Description:
// Owner: Dragan Milenkovic <dragan@ponos.technology>
//////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "rust/cxx.h"
#include "ponos-engine/src/engine.rs.h"

#include <memory>
#include <cstdint>
#include <array>

namespace ponos {

    void run_fpga_msm_vitis_mont(rust::Slice<const RawScalar256> scalars, rust::Slice<const RawArkAffine> bases, RawArkAffine & result);
    void run_fpga_msm_vitis_normal(rust::Slice<const RawScalar256> scalars, rust::Slice<const NBase384> bases, RawArkAffine & result);

}
