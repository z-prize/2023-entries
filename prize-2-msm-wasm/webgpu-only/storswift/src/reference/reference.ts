import { BigIntPoint, U32ArrayPoint } from "./types";
import { FieldMath } from "./utils/FieldMath";
import { loadWasmModule } from "./wasm-loader/wasm-loader";
import { naive_msm } from "./webgpu/entries/naiveMSMEntry";
import { bigIntsToU16Array, bigIntsToU32Array } from "./webgpu/utils";
import { pippinger_msm } from "./webgpu/entries/pippengerMSMEntry";
import { wasmMSM } from "./workers/wasmMSM";

export const webgpu_pippenger_msm = async (
    baseAffinePoints: BigIntPoint[] | U32ArrayPoint[] | Uint32Array,
    scalars: bigint[] | Uint32Array[]| Uint32Array
) => {
  const fieldMath = new FieldMath();
  const pointsAsU32s = (baseAffinePoints as BigIntPoint[]).map(point => fieldMath.createPoint(point.x, point.y, point.t, point.z));
  const scalarsAsU16s = Array.from(bigIntsToU16Array(scalars as bigint[]));
  return await pippinger_msm(pointsAsU32s, scalarsAsU16s, fieldMath);
}

export const webgpu_compute_msm = async (
    baseAffinePoints: BigIntPoint[] | U32ArrayPoint[] | Uint32Array,
    scalars: bigint[] | Uint32Array[]| Uint32Array
): Promise<{x: bigint, y: bigint}> => {
  const flattenedPoints = (baseAffinePoints as BigIntPoint[]).flatMap(point => [point.x, point.y]);
  const pointsAsU32s = bigIntsToU32Array(flattenedPoints);
  const scalarsAsU32s = bigIntsToU32Array(scalars as bigint[]);
  return await naive_msm(pointsAsU32s, scalarsAsU32s);
};

export const wasm_compute_msm = async (
    baseAffinePoints: BigIntPoint[] | U32ArrayPoint[]| Uint32Array,
    scalars: bigint[] | Uint32Array[]| Uint32Array
): Promise<{x: bigint, y: bigint}> => {
  const aleo = await loadWasmModule();
  const groups = (baseAffinePoints as BigIntPoint[]).map(point => point.x.toString() + 'group');
  const aleoScalars = (scalars as bigint[]).map(scalar => scalar.toString() + 'scalar');
  const xresult = aleo.Address.msm(groups, aleoScalars);
  const xNum = xresult.slice(0, xresult.indexOf('group'));
  return new FieldMath().getPointFromX(BigInt(xNum));
};

export const webgpu_best_msm = async (
    baseAffinePoints: BigIntPoint[] | U32ArrayPoint[] | Uint32Array,
    scalars: bigint[] | Uint32Array[]| Uint32Array
): Promise<{x: bigint, y: bigint}> => {
  return await webgpu_pippenger_msm(baseAffinePoints, scalars);
};

export const wasm_compute_msm_parallel = async (
    baseAffinePoints: BigIntPoint[] | U32ArrayPoint[] | Uint32Array,
    scalars: bigint[] | Uint32Array[]| Uint32Array
): Promise<{x: bigint, y: bigint}> => {
  const groups = (baseAffinePoints as BigIntPoint[]).map(point => point.x.toString() + 'group');
  const aleoScalars = (scalars as bigint[]).map(scalar => scalar.toString() + 'scalar');
  const xNum = await wasmMSM(groups, aleoScalars);
  return new FieldMath().getPointFromX(BigInt(xNum));
};
