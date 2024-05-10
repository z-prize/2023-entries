/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Antony Suresh
            Sougata Bhattacharya 
            Niall Emmart

***/

import { BigIntPoint, U32ArrayPoint } from "../reference/types";
import { WasmModule } from "./submission-types";

/* eslint-disable @typescript-eslint/no-unused-vars */

let msmEngineInternal: any = null;
export const compute_msm = async (
  baseAffinePoints: BigIntPoint[] | U32ArrayPoint[] | Buffer,
  scalars: bigint[] | Uint32Array[] | Buffer
): Promise<{ x: bigint, y: bigint }> => {
  const cores = Math.min(navigator.hardwareConcurrency, 20);
  const memPages = 2600;
  const wasmBytes = await fetchWASM('msm.wasm');
  const myEngine = await msmEngine(wasmBytes, memPages, cores);
  const workers = myEngine.workers;
  console.log("Engine setup complete");
  const memoryU32 = new Uint32Array(myEngine.sharedMemory.buffer, 0, memPages * 16384);
  const start = performance.now();
  const pointsBuffer = baseAffinePoints as Buffer;
  let pointCount = pointsBuffer.length/96;
  pointCount = copyPoints(memoryU32, myEngine.pointsOffset,pointsBuffer);
  const scalarBuffer = scalars as Buffer;
  copyScalars(memoryU32, myEngine.scalarsOffset,scalarBuffer);

  const end = performance.now();
  console.log(end - start);
  const exports = myEngine.wasmInstance.exports as unknown as WasmModule;
  exports.initializeCounters(pointCount);
  for (let i = 0; i < workers.length; i++)
    workers[i].postMessage({ module: myEngine.wasmModule, sharedMemory: myEngine.sharedMemory, workerID: i });
  return new Promise((resolve) => {
    workers[0].onmessage = (message:any) => {
      resolve({ x: extract384(memoryU32, myEngine.resultOffset), y: extract384(memoryU32, myEngine.resultOffset + 12) });
    }
  });
};

async function msmEngine(wasmBytes: ArrayBuffer, memPages: number, workerCount: number) {
  if (msmEngineInternal != null) return msmEngineInternal;
  
  const sharedMemory = new WebAssembly.Memory({ initial: memPages, maximum: memPages, shared: true });
  const imports = { env: { memory: sharedMemory } };

  const wasmModule = await WebAssembly.compile(wasmBytes);
  const wasmInstance = await WebAssembly.instantiate(wasmModule, imports);
  const workers = [];
  const exports = wasmInstance.exports as unknown as WasmModule;
  console.log("Creating ", workerCount, "workers to the job");
  for (let i = 0; i < workerCount; i++) {
    workers.push(new Worker('worker.js'));
  }

  const resultOffset: number = exports.resultAddress() >> 2;
  console.log("resultOffset: " + resultOffset)

  const pointsOffset: number = exports.pointsAddress() >> 2;
  console.log("pointsOffset: " + pointsOffset)

  const scalarsOffset: number = exports.scalarsAddress() >> 2;
  console.log("scalarsOffset: " + scalarsOffset)

  msmEngineInternal = {
    sharedMemory: sharedMemory,
    wasmBytes: wasmBytes,
    wasmModule: wasmModule,
    wasmInstance: wasmInstance,
    resultOffset: resultOffset,
    pointsOffset: pointsOffset,
    scalarsOffset: scalarsOffset,
    workers: workers
  };

  return msmEngineInternal;
}

async function fetchWASM(urlPath: string) {
  const response = await fetch(urlPath);
  return response.arrayBuffer();
}


function insert256(u32Array: Uint32Array, x: bigint, offset: number) {
  const mask = BigInt(0xFFFFFFFFn), shift = BigInt(32);

  for (let i = 0; i < 8; i++) {
    u32Array[offset + i] = Number(x & mask);
    x = x >> shift;
  }
}

function extract256(u32Array: Uint32Array, offset: number) {
  const shift = BigInt(32);
  let x;

  x = BigInt(u32Array[offset + 7]);
  for (let i = 6; i >= 0; i--) {
    x = x << shift;
    x = x + BigInt(u32Array[offset + i]);
  }
  return x;
}

function insert384(u32Array: Uint32Array, x: bigint, offset: any) {
  const mask = 0xFFFFFFFFn;
  for (let i = 0; i < 12; i++) {
    u32Array[offset + i] = Number(x & mask);
    x = x >> 32n;
  }
}

function extract384(u32Array: Uint32Array, offset: any) {
  let x: bigint;
  x = BigInt(u32Array[offset + 11]);
  for (let i = 10; i >= 0; i--) {
    x = x << 32n;
    x = x + BigInt(u32Array[offset + i]);
  }
  return x.valueOf();
}

function copyPoints(wasmArray: Uint32Array, wasmOffset: number, buffer: Buffer) {
  const pointCount = buffer.length / 96; // ( 384 / 8 ) * 2
  console.log("points",pointCount);
  const sourcePoints = new Uint32Array(buffer.buffer, 0, pointCount * 2 * 12);
  for( let i = 0; i < pointCount * 24; i++) {
    wasmArray[wasmOffset + i] = sourcePoints[i];
  }
  return pointCount;
}

function copyScalars(wasmArray:Uint32Array, wasmOffset: number, buffer: Buffer) {
  const scalarCount = buffer.length / 32;
  const sourceScalars = new Uint32Array(buffer.buffer, 0 , scalarCount * 8);
  for( let i=0; i<scalarCount*8; i++){
    wasmArray[wasmOffset+i] = sourceScalars[i];
  }
}

