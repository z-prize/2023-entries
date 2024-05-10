/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Antony Suresh
            Sougata Bhattacharya 
            Niall Emmart

***/

export interface WasmModule {
    initializeCounters: (a: number) => void;
    resultAddress: () => any;
    pointsAddress: () => any;
    scalarsAddress: () => any;
    // Define other exported functions here
}