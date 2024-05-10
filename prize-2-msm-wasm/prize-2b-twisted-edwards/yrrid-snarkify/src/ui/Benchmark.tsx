/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useState } from 'react';
import { BigIntPoint, U32ArrayPoint } from '../reference/types';
import { CheckIcon } from '@heroicons/react/20/solid';
import { XMarkIcon } from '@heroicons/react/20/solid';

interface BenchmarkProps {
  bold?: boolean;
  disabled: boolean;
  name: string;
  baseAffinePoints: BigIntPoint[] | U32ArrayPoint[] | Buffer;
  scalars: bigint[] | Uint32Array[] | Buffer;
  expectedResult: {x: bigint, y: bigint} | null;
  msmFunc: (
    baseAffinePoints: BigIntPoint[] | U32ArrayPoint[] | Buffer,
    scalars: bigint[] | Uint32Array[] | Buffer
    ) => Promise<{x: bigint, y: bigint}>
  postResult: (result: {x: bigint, y: bigint}, timeMS: number, msmFunc: string) => void;
}

export const Benchmark: React.FC<BenchmarkProps> = (
  {bold, disabled, name, baseAffinePoints, scalars, expectedResult, msmFunc, postResult}
  ) => {
  const [runTime, setRunTime] = useState(0);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<{x: bigint, y: bigint}>({x: BigInt(0), y: BigInt(0)});

  const runFunc = async () => {
    setRunning(true);
    const start = performance.now();
    const result = await msmFunc(baseAffinePoints, scalars);
    const end = performance.now();
    setRunTime(end - start);
    setResult(result);
    postResult(result, end - start, name);
    setRunning(false);
    return result;
  };

  const correctnessMark = (result: {x: bigint, y: bigint} | null, expectedResult: {x: bigint, y: bigint} | null) => {
    if (result && expectedResult) {
      if (result.x.toString() === expectedResult.x.toString() && result.y.toString() === expectedResult.y.toString()) {
        return <CheckIcon className="h-4 w-4 text-green-500" aria-hidden="true" />;
      } else {
        return <XMarkIcon className="h-4 w-4 text-red-500" aria-hidden="true" />;
      }
    }

    return <></>;
  };

  const spin = () => <div className="w-4 h-4 border-t-2 border-white rounded-full animate-spin"></div>;

  return (
    <>
      <div className="flex items-center space-x-4 px-5">
        <div className={`text-gray-800 w-40 px-2 ${bold ? 'font-bold' : ''}`}>{name}</div> 
        <button disabled={disabled} className="bg-blue-500 hover:bg-blue-700 text-white px-4 py-2 rounded-md"  onClick={async () => { await runFunc()}}>
          {running || disabled ? spin() : 'Compute'}
        </button>
        <div className="text-gray-800 w-36 truncate">{runTime > 0 ? `${runTime} ms` : 'Run Time: 0ms'}</div>
        {correctnessMark(result, expectedResult)}
        <div className="text-gray-800 w-36">x: {result.x.toString()} y: {result.y.toString()}</div>
      </div>
      <hr className='p-2'/>
    </>
  );
};