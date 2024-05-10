/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Antony Suresh
            Sougata Bhattacharya
            Niall Emmart

***/

var instantiation=null;

async function wasmInstantiate(message) {
  if(instantiation===null) {
    const data=message.data;
    const module=data.module;
    const sharedMemory=data.sharedMemory;
    const workerID=message.data.workerID;
    var   sp, hb;

    instantiation=await WebAssembly.instantiate(module, {env: {memory: sharedMemory}});
    sp=instantiation.exports.__stack_pointer;
    hb=instantiation.exports.__heap_base;
    sp.value=hb.value + 65536 + 65536*workerID
  }
  return instantiation;
}

onmessage = (message) => {
  const workerID=message.data.workerID;
  wasmInstantiate(message).then(wasm => {
    wasm.exports.computeMSM();
    if(workerID==0)
      postMessage('msm done');
  });
};
