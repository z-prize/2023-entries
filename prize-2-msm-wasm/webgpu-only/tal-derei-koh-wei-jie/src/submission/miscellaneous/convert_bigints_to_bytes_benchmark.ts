import { BigIntPoint, U32ArrayPoint } from "../../reference/types";
import { bigints_to_u8_for_gpu } from "../implementation/cuzk/utils";
import { bigints_to_16_bit_words_for_gpu } from "../miscellaneous/utils";
import assert from "assert";

const num_words = 16;
const word_size = 16;

export const convert_bigints_to_bytes_benchmark = async (
  baseAffinePoints: BigIntPoint[] | U32ArrayPoint[] | Buffer,
  scalars: bigint[] | Uint32Array[] | Buffer,
): Promise<{ x: bigint; y: bigint }> => {
  //let code = ''
  //for (let i = 0; i < 16; i ++) {
  ////code += `    shift = BigInt(${(15 - i) * 16})\n`
  //code += `    result[i * 64 + ${(15 - i) * 4}] = Number((val >> BigInt(${(15 - i) * 16})) & mask)\n`
  //code += `    result[i * 64 + ${(15 - i) * 4 + 1}] = Number((val >> (BigInt(${(15 - i) * 16 + 8}))) & mask)\n\n`
  //}
  //console.log(code)

  const start_0 = Date.now();
  const x_y_coords_bytes = bigints_to_u8_for_gpu(scalars as bigint[]);
  const elapsed_0 = Date.now() - start_0;
  console.log(
    `bigints_to_u8_for_gpu took ${elapsed_0}ms to convert ${scalars.length} scalars`,
  );

  const start_1 = Date.now();
  const x_y_coords_bytes_2 = bigints_to_16_bit_words_for_gpu(
    scalars as bigint[],
  );
  const elapsed_1 = Date.now() - start_1;
  console.log(
    `bigints_to_16_bit_words_for_gpu took ${elapsed_1}ms to convert ${scalars.length} scalars`,
  );

  assert(x_y_coords_bytes.length == x_y_coords_bytes_2.length);
  for (let i = 0; i < x_y_coords_bytes.length; i++) {
    assert(x_y_coords_bytes[i] == x_y_coords_bytes_2[i]);
  }
  return { x: BigInt(0), y: BigInt(0) };
};
