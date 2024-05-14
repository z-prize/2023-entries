#!/bin/bash

if [ -z "$U250_DEVICE" ]; then
    echo "ERROR: Environment variable U250_DEVICE is not set."
    exit 1
fi

sudo bash -c "source /opt/xilinx/xrt/setup.sh && xbmgmt program --shell /lib/firmware/xilinx/12c8fafb0632499db1c0c6676271b8a6/partition.xsabin -d $U250_DEVICE"
