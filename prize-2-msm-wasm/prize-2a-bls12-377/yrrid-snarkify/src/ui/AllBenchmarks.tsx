import React, { useEffect, useState } from 'react';
import { Benchmark } from './Benchmark';
import { bigIntToU32Array, bigIntsToBufferLE, generateRandomFields } from '../reference/webgpu/utils';
import { BigIntPoint, U32ArrayPoint } from '../reference/types';
import { wasm_compute_bl12_377_msm_buffer, wasm_compute_bls12_377_msm, wasm_compute_msm,  wasm_compute_msm_parallel } from '../reference/reference';
import { compute_msm } from '../submission/submission';
import CSVExportButton from './CSVExportButton';
import { TestCaseDropDown } from './TestCaseDropDown';
import { PowersTestCase, TestCase, loadTestCase } from '../test-data/testCases';

export const AllBenchmarks: React.FC = () => {
  const initialDefaultInputSize = 1_000;
  const [inputSize, setInputSize] = useState(initialDefaultInputSize);
  const [power, setPower] = useState<string>('2^0');
  const [inputSizeDisabled, setInputSizeDisabled] = useState(false);
  const [baseAffineBigIntPoints, setBaseAffineBigIntPoints] = useState<BigIntPoint[]>([]);
  const [bigIntScalars, setBigIntScalars] = useState<bigint[]>([]);
  const [u32Points, setU32Points] = useState<U32ArrayPoint[]>([]);
  const [u32Scalars, setU32Scalars] = useState<Uint32Array[]>([]);
  const [bufferPoints, setBufferPoints] = useState<Buffer>(Buffer.alloc(0));
  const [bufferScalars, setBufferScalars] = useState<Buffer>(Buffer.alloc(0));
  const [expectedResult, setExpectedResult] = useState<{x: bigint, y: bigint} | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [benchmarkResults, setBenchmarkResults] = useState<any[][]>([["InputSize", "MSM Func", "Time (MS)"]]);
  const [comparisonResults, setComparisonResults] = useState<{ x: bigint, y: bigint, timeMS: number, msmFunc: string, inputSize: number }[]>([]);
  const [disabledBenchmark, setDisabledBenchmark] = useState<boolean>(false);

  const postResult = (result: {x: bigint, y: bigint}, timeMS: number, msmFunc: string) => {
    const benchMarkResult = [inputSizeDisabled ? power : inputSize, msmFunc, timeMS];
    setBenchmarkResults([...benchmarkResults, benchMarkResult]);
    setComparisonResults([...comparisonResults, {x: result.x, y: result.y, timeMS, msmFunc, inputSize}]);
    if (msmFunc.includes('Aleo Wasm')) {
      setExpectedResult(result);
    }
  };

  const loadAndSetData = async (power: PowersTestCase) => {
    setInputSizeDisabled(true);
    setInputSize(0);
    setDisabledBenchmark(true);
    setPower(`2^${power}`);
    const testCase = await loadTestCase(power);
    setTestCaseData(testCase);
  }

  const setTestCaseData = async (testCase: TestCase) => {
    setBaseAffineBigIntPoints(testCase.baseAffinePoints);
    const newU32Points = testCase.baseAffinePoints.map((point) => {
      return {
        x: bigIntToU32Array(point.x, 384),
        y: bigIntToU32Array(point.y, 384),
        z: bigIntToU32Array(point.z, 384),
      }});
    setU32Points(newU32Points);

    const xyArray: bigint[] = [];
    testCase.baseAffinePoints.map((point) => {
      xyArray.push(point.x);
      xyArray.push(point.y);
    });
    const pointsBufferLE = bigIntsToBufferLE(xyArray, 384);
    setBufferPoints(pointsBufferLE);
    setBigIntScalars(testCase.scalars);
    const newU32Scalars = testCase.scalars.map((scalar) => bigIntToU32Array(scalar));
    setU32Scalars(newU32Scalars);
    const scalarBuffer = bigIntsToBufferLE(testCase.scalars, 256);
    setBufferScalars(scalarBuffer);
    setExpectedResult(testCase.expectedResult);
    setDisabledBenchmark(false);
  };

  const useRandomInputs = () => {
    setDisabledBenchmark(true);
    setInputSizeDisabled(false);
    setExpectedResult(null);
    setInputSize(initialDefaultInputSize);
    setDisabledBenchmark(false);
  };

  useEffect(() => {
    async function generateNewInputs() {
      // creating random points is slow, so for now use a single fixed base.
      const x = BigInt('111871295567327857271108656266735188604298176728428155068227918632083036401841336689521497731900230387779623820740');
      const y = BigInt('76860045326390600098227152997486448974650822224305058012700629806287380625419427989664237630603922765089083164740');
      const z = BigInt('1');
      const point: BigIntPoint = {x, y, z};
      const newPoints = Array(inputSize).fill(point);
      setBaseAffineBigIntPoints(newPoints);
      const newU32Points = newPoints.map((point) => {
        return {
          x: bigIntToU32Array(point.x, 384),
          y: bigIntToU32Array(point.y, 384),
          z: bigIntToU32Array(point.z, 384),
        }});
      setU32Points(newU32Points);
      const xyArray: bigint[] = [];
      newPoints.map((point) => {
        xyArray.push(point.x);
        xyArray.push(point.y);
      });
      const pointsBufferLE = bigIntsToBufferLE(xyArray, 384);
      setBufferPoints(pointsBufferLE);

      const newScalars = generateRandomFields(inputSize);
      setBigIntScalars(newScalars);
      const newU32Scalars = newScalars.map((scalar) => bigIntToU32Array(scalar));
      setU32Scalars(newU32Scalars);
      const scalarBuffer = bigIntsToBufferLE(newScalars, 256);
      setBufferScalars(scalarBuffer);
    }
    generateNewInputs();
    setComparisonResults([]);
  }, [inputSize]);
  
  return (
    <div>
      <div className="flex items-center space-x-4 px-5">
        <div className="text-gray-800">Input Size:</div>
        <input
          type="text"
          className="w-24 border border-gray-300 rounded-md px-2 py-1"
          value={inputSize}
          disabled={inputSizeDisabled}
          onChange={(e) => setInputSize(parseInt(e.target.value))}
        />
        <TestCaseDropDown useRandomInputs={useRandomInputs} loadAndSetData={loadAndSetData}/>
        <CSVExportButton data={benchmarkResults} filename={'msm-benchmark'} />
      </div>
      
      <Benchmark
        name={'Aleo Wasm: Single Thread'}
        disabled={disabledBenchmark}
        baseAffinePoints={baseAffineBigIntPoints}
        scalars={bigIntScalars}
        expectedResult={expectedResult}
        msmFunc={wasm_compute_bls12_377_msm}
        postResult={postResult}
      />
      <Benchmark
        name={'Aleo Wasm: Buffer Input'}
        disabled={disabledBenchmark}
        baseAffinePoints={bufferPoints}
        scalars={bufferScalars}
        expectedResult={expectedResult}
        msmFunc={wasm_compute_bl12_377_msm_buffer}
        postResult={postResult}
      />
      <Benchmark
        name={'Your MSM'}
        disabled={disabledBenchmark}
        baseAffinePoints={bufferPoints}
        scalars={bufferScalars}
        expectedResult={expectedResult}
        msmFunc={compute_msm}
        postResult={postResult}
        bold={true}
      />
    </div>
  )
};