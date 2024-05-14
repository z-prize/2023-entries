#!/bin/bash

source /opt/xilinx/xrt/setup.sh
xbmgmt program --shell /lib/firmware/xilinx/12c8fafb0632499db1c0c6676271b8a6/partition.xsabin -d 01:00.0
