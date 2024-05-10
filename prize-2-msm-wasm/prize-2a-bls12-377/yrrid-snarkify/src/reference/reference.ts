import { BigIntPoint, U32ArrayPoint } from "./types";
import { FieldMath } from "./utils/FieldMath";
import { readBigIntsFromBufferLE } from "./webgpu/utils";
import { wasmMSM } from "./workers/wasmMSM";
import * as Aleo from "@demox-labs/gpu-wasm-expose";

export const wasm_compute_msm = async (
  baseAffinePoints: BigIntPoint[] | U32ArrayPoint[] | Buffer,
  scalars: bigint[] | Uint32Array[] | Buffer
  ): Promise<{x: bigint, y: bigint}> => {
  const groups = (baseAffinePoints as BigIntPoint[]).map(point => point.x.toString() + 'group');
  const aleoScalars = (scalars as bigint[]).map(scalar => scalar.toString() + 'scalar');
  const xresult = Aleo.Address.msm(groups, aleoScalars);
  const xNum = xresult.slice(0, xresult.indexOf('group'));
  return new FieldMath().getPointFromX(BigInt(xNum));
};

export const wasm_compute_bls12_377_msm = async (
  baseAffinePoints: BigIntPoint[] | U32ArrayPoint[] | Buffer,
  scalars: bigint[] | Uint32Array[] | Buffer
  ): Promise<{x: bigint, y: bigint}> => {
    const aleoXs = (baseAffinePoints as BigIntPoint[]).map(point => point.x.toString());
    const aleoYs = (baseAffinePoints as BigIntPoint[]).map(point => point.y.toString());
    const affineResult = Aleo.Address.bls12_377_msm(aleoXs, aleoYs, (scalars as bigint[]).map(scalar => scalar.toString()));
    // affineResult will be in the form Affine(x=..., y=...), so need to parse
    const xresult = affineResult.slice(affineResult.indexOf('x=') + 2, affineResult.indexOf(', y='));
    const yresult = affineResult.slice(affineResult.indexOf('y=') + 2, affineResult.indexOf(')'));
    return { x: BigInt(xresult), y: BigInt(yresult) };
};

export const wasm_compute_msm_parallel = async (
  baseAffinePoints: BigIntPoint[] | U32ArrayPoint[] | Buffer,
  scalars: bigint[] | Uint32Array[] | Buffer
  ): Promise<{x: bigint, y: bigint}> => {
  const groups = (baseAffinePoints as BigIntPoint[]).map(point => point.x.toString() + 'group');
  const aleoScalars = (scalars as bigint[]).map(scalar => scalar.toString() + 'scalar');
  const xNum = await wasmMSM(groups, aleoScalars);
  return new FieldMath().getPointFromX(BigInt(xNum));
};

export const wasm_compute_bl12_377_msm_buffer = async (
  baseAffinePoints: BigIntPoint[] | U32ArrayPoint[] | Buffer,
  scalars: bigint[] | Uint32Array[] | Buffer
  ): Promise<{x: bigint, y: bigint}> => {
    const scalarsArray = readBigIntsFromBufferLE(scalars as Buffer, 256);

    const xypoints = readBigIntsFromBufferLE(baseAffinePoints as Buffer, 384);
    const aleoXs: string[] = [];
    const aleoYs: string[] = [];
    xypoints.map((num, index) => {
      if (index % 2 === 0) {
        aleoXs.push(num.toString());
      } else {
        aleoYs.push(num.toString());
      }
    });
    const affineResult = Aleo.Address.bls12_377_msm(aleoXs, aleoYs, (scalarsArray).map(scalar => scalar.toString()));
    // affineResult will be in the form Affine(x=..., y=...), so need to parse
    const xresult = affineResult.slice(affineResult.indexOf('x=') + 2, affineResult.indexOf(', y='));
    const yresult = affineResult.slice(affineResult.indexOf('y=') + 2, affineResult.indexOf(')'));
    return { x: BigInt(xresult), y: BigInt(yresult) };
};
