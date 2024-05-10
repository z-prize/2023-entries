import { ALEO_FIELD_MODULUS } from "../params/AleoConstants";
import { BigIntPoint } from "../types";

export interface gpuU32Inputs {
  u32Inputs: Uint32Array;
  individualInputSize: number;
}

export const bigIntsToU16Array = (beBigInts: bigint[]): Uint16Array => {
  const intsAs16s = beBigInts.map(bigInt => bigIntToU16Array(bigInt));
  const u16Array = new Uint16Array(beBigInts.length * 16);
  intsAs16s.forEach((intAs16, index) => {u16Array.set(intAs16, index * 16)});
  return u16Array;
}

export const bigIntToU16Array = (beBigInt: bigint): Uint16Array => {
  const numBits = 384;
  const bitsPerElement = 16;
  const numElements = numBits / bitsPerElement;
  const u16Array = new Uint16Array(numElements);
  const mask = (BigInt(1) << BigInt(bitsPerElement)) - BigInt(1); // Create a mask for the lower 32 bits

  let tempBigInt = beBigInt;
  for (let i = numElements - 1; i >= 0; i--) {
    u16Array[i] = Number(tempBigInt & mask); // Extract the lower 32 bits
    tempBigInt >>= BigInt(bitsPerElement); // Right-shift the remaining bits
  }

  return u16Array;
};

export const flattenU32s = (u32Arrays: Uint32Array[]): Uint32Array => {
  const flattenedU32s = new Uint32Array(u32Arrays.length * u32Arrays[0].length);
  u32Arrays.forEach((u32Array, index) => {
    flattenedU32s.set(u32Array, index * u32Array.length);
  });
  return flattenedU32s;
};

// assume bigints are big endian 384-bit integers
export const bigIntsToU32Array = (beBigInts: bigint[], bigIntSize = 256): Uint32Array => {
  const intsAs32s = beBigInts.map(bigInt => bigIntToU32Array(bigInt, bigIntSize));
  const u32Array = new Uint32Array(beBigInts.length * bigIntSize / 32);
  intsAs32s.forEach((intAs32, index) => {u32Array.set(intAs32, index * 8)});
  return u32Array;
};

export const bigIntToU32Array = (beBigInt: bigint, bigIntSize = 256): Uint32Array => {
  const bitsPerElement = 32;
  const numElements = bigIntSize / bitsPerElement;
  const u32Array = new Uint32Array(numElements);
  const mask = (BigInt(1) << BigInt(bitsPerElement)) - BigInt(1); // Create a mask for the lower 32 bits

  let tempBigInt = beBigInt;
  for (let i = numElements - 1; i >= 0; i--) {
    u32Array[i] = Number(tempBigInt & mask); // Extract the lower 32 bits
    tempBigInt >>= BigInt(bitsPerElement); // Right-shift the remaining bits
  }

  return u32Array;
};

export const bigIntBufferLE = (bigInt: bigint, bigIntSize = 256): Buffer => {
  const hexString = bigInt.toString(16).padStart(bigIntSize / 4, '0');
  const bytes = Buffer.from(hexString, 'hex');
  return bytes.reverse();
};

export const bigIntsToBufferLE = (bigInts: bigint[], bigIntSize = 256): Buffer => {
  const bigIntBuffers = bigInts.map(bigInt => bigIntBufferLE(bigInt, bigIntSize));
  return Buffer.concat(bigIntBuffers);
};

export const readBigIntsFromBufferLE = (buffer: Buffer, bigIntSize = 256): bigint[] => {
  const totalBigInts = buffer.length / (bigIntSize / 8);
  const bigInts: bigint[] = [];
  for (let i = 0; i < totalBigInts; i++) {
    // references original buffer memory
    const singleBigIntBuffer = buffer.slice(i * (bigIntSize / 8), (i + 1) * (bigIntSize / 8));
    const copyBuffer = Buffer.from(singleBigIntBuffer);
    const bigIntBufferString = copyBuffer.reverse().toString('hex');
    const singleBigInt = BigInt('0x' + bigIntBufferString);
    bigInts.push(singleBigInt);
  }

  return bigInts;
}

export const u32ArrayToBigInts = (u32Array: Uint32Array, bigIntSize = 256): bigint[] => {
  const bigInts = [];
  const bitsPerElement = 32;
  const chunkSize = bigIntSize / bitsPerElement;

  for (let i = 0; i < u32Array.length; i += chunkSize) {
    let bigInt = BigInt(0);
    for (let j = 0; j < chunkSize; j++) {
      if (i + j >= u32Array.length) break; // Avoid out-of-bounds access
      const u32 = BigInt(u32Array[i + j]);
      bigInt |= (u32 << (BigInt(chunkSize - 1 - j) * BigInt(bitsPerElement)));
    }
    bigInts.push(bigInt);
  }

  return bigInts;
};

export const generateRandomFields = (inputSize: number): bigint[] => {
  const randomBigInts = [];
  for (let i = 0; i < inputSize; i++) {
    randomBigInts.push(createRandomAleoFieldInt());
  }

  return randomBigInts;
};

export const convertBigIntsToWasmFields = (bigInts: bigint[]): string[] => {
  return bigInts.map(bigInt => bigInt.toString() + 'field');
};

const createRandomAleoFieldInt = () => {
  let bigIntString = '';
  for (let i = 0; i < 8; i++) {
    bigIntString += Math.floor(Math.random() * (2**32 - 1));
  }
  return BigInt(bigIntString) % ALEO_FIELD_MODULUS;
}

export const stripFieldSuffix = (field: string): string => {
  return field.slice(0, field.length - 5);
};

export const stripGroupSuffix = (group: string): string => {
  return group.slice(0, group.length - 5);
};

export const chunkArray = (inputsArray: gpuU32Inputs[], batchSize: number): gpuU32Inputs[][] => {
  let index = 0;
  const chunkedArray: gpuU32Inputs[][] = [];
  const firstInputLength = inputsArray[0].u32Inputs.length / inputsArray[0].individualInputSize;

  while (index < firstInputLength) {
      const newIndex = index + batchSize;
      const tempArray: gpuU32Inputs[] = [];
      inputsArray.forEach(bufferData => {
        const chunkedGpuU32Inputs = bufferData.u32Inputs.slice(index * bufferData.individualInputSize, newIndex * bufferData.individualInputSize);
        tempArray.push({
          u32Inputs: chunkedGpuU32Inputs,
          individualInputSize: bufferData.individualInputSize
        });
      });
      index = newIndex;
      chunkedArray.push(tempArray);
  }

  return chunkedArray;
};

export function concatUint32Arrays(array1: Uint32Array, array2: Uint32Array): Uint32Array {
  // Create a new Uint32Array with a length equal to the sum of the lengths of array1 and array2
  const result = new Uint32Array(array1.length + array2.length);

  // Copy the elements from array1 into the new array
  result.set(array1, 0);

  // Copy the elements from array2 into the new array, starting at the index after the last element of array1
  result.set(array2, array1.length);

  return result;
}