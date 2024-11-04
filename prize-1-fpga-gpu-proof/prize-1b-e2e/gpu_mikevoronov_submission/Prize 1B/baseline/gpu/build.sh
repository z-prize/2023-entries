#!/bin/bash

(cd "gpu/TrapdoorTech-zprize-crypto-cuda/src/cub" ; ./compile.sh)
(cd "gpu/matter_gpu_msm" ; ./compile.sh)
(cd "gpu/yarrid-gpu-msm" ; ./compile.sh)
