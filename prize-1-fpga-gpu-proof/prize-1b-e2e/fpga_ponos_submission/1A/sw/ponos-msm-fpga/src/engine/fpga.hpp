#pragma once

#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

namespace ponos {

    template <typename T>
    struct XrtBoAndPtr {
        xrt::bo buffer_object;
        T * ptr = nullptr;
    };

}
