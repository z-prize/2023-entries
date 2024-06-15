import { BigIntPoint, U32ArrayPoint } from "../../reference/types";
import { bigIntsToBufferLE } from "../../reference/webgpu/utils";
import { PowersTestCase, loadTestCase } from "../../test-data/testCases";
import { compute_msm } from "../submission";

export const full_benchmarks = async (
  {}: BigIntPoint[] | U32ArrayPoint[] | Buffer,
  {}: bigint[] | Uint32Array[] | Buffer,
): Promise<{ x: bigint; y: bigint }> => {
  const DELAY = 100;
  const NUM_RUNS = 5;

  // Inclusive
  const START_POWER: PowersTestCase = 16;
  const END_POWER: PowersTestCase = 20;

  const all_results: any = {};
  console.log(
    `Running benchmarks for powers ${START_POWER} to ${END_POWER} (inclusive)`,
  );

  let do_recompile = true;

  // Load testcases in advance
  const testcases: any = {};
  for (let power = START_POWER; power <= END_POWER; power++) {
    const testcase = await loadTestCase(power);
    const xyArray: bigint[] = [];
    testcase.baseAffinePoints.map((point: any) => {
      xyArray.push(point.x);
      xyArray.push(point.y);
    });

    const bufferPoints = bigIntsToBufferLE(xyArray, 256);
    const bufferScalars = bigIntsToBufferLE(testcase.scalars, 256);
    testcases[power] = {
      bufferPoints,
      bufferScalars,
      expectedResult: testcase.expectedResult,
    };
  }

  for (let power = START_POWER; power <= END_POWER; power++) {
    console.log(
      `Running ${
        NUM_RUNS + 1
      } invocations of cuzk_gpu() for 2^${power} inputs, please wait...`,
    );
    const testcase = testcases[power];

    /*
      Iterate test cases of powers 16 to 20 inclusive
      For each test case:
        - inject some random variables into each shader to force recompilation
        - measure the first run
        - measure NUM_RUNS more runs
        - report the time taken for all runs, the average, and the average
          excluding the first (compile) run in JSON and Markdown
    */

    // Measure the first run, which forces a recompile
    const first_run_start = Date.now();
    const msm = await compute_msm(
      testcase.bufferPoints,
      testcase.bufferScalars,
      false,
      do_recompile,
    );

    // Only recompile on the 1st run, regardless of power.
    // This should simulate the performance during judging more accurately.
    do_recompile = false;

    const first_run_elapsed = Date.now() - first_run_start;

    const expected = testcase.expectedResult;
    if (msm.x !== expected.x || msm.y !== expected.y) {
      console.error(
        `WARNING: the result of cuzk_gpu is incorrect for 2^${power}`,
      );
    }

    await delay(DELAY);

    const results: {
      first_run_elapsed: number;
      subsequent_runs: number[];
      full_average: number;
      subsequent_average: number;
    } = {
      first_run_elapsed,
      subsequent_runs: [],
      full_average: 0,
      subsequent_average: 0,
    };

    for (let i = 0; i < NUM_RUNS; i++) {
      const start = Date.now();
      // Run without forcing a recompile
      await compute_msm(
        testcase.bufferPoints,
        testcase.bufferScalars,
        false,
        false,
      );
      const elapsed = Date.now() - start;

      results.subsequent_runs.push(elapsed);
      await delay(DELAY);
    }

    let subsequent_total = 0;
    for (let i = 0; i < results.subsequent_runs.length; i++) {
      subsequent_total += results.subsequent_runs[i];
    }

    results["full_average"] = Math.round(
      (results.first_run_elapsed + subsequent_total) /
        (1 + results.subsequent_runs.length),
    );

    results["subsequent_average"] = Math.round(
      subsequent_total / results.subsequent_runs.length,
    );

    all_results[power] = results;
  }

  let header = `| MSM size | 1st run |`;
  for (let i = 0; i < NUM_RUNS; i++) {
    header += ` Run ${i + 1} |`;
  }

  header += ` Average (incl 1st) | Average (excl 1st) |`;
  header += `\n|-|-|-|-|`;
  for (let i = 0; i < NUM_RUNS; i++) {
    header += `-|`;
  }
  header += `\n`;
  let body = ``;

  for (let power = START_POWER; power <= END_POWER; power++) {
    const result = all_results[power];
    let md = `| 2^${power} | \`${result.first_run_elapsed}\` |`;
    for (let i = 0; i < result.subsequent_runs.length; i++) {
      const r = result.subsequent_runs[i];
      md += ` \`${r}\` |`;
    }

    body +=
      md +
      ` **\`${result.full_average}\`** | **\`${result.subsequent_average}\`** |\n`;
  }

  console.log(header + body.trim());

  if (START_POWER < END_POWER) {
    console.log(
      `Note that shaders were forced to recompile only for MSM size 2^${START_POWER}`,
    );
  }
  return { x: BigInt(0), y: BigInt(1) };
};

export const delay = (duration: number) => {
  return new Promise((resolve) => setTimeout(resolve, duration));
};
