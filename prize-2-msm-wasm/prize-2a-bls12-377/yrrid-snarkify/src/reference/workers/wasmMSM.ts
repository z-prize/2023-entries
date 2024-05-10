import { spawn, Thread, Worker } from 'threads';

export const wasmMSM = async (groups: string[], scalars: string[]): Promise<string> => {
  const wasmMSMWorker = await spawn(new Worker('./wasmMSM.js'));
  const xNum = await wasmMSMWorker(groups, scalars);
  await Thread.terminate(wasmMSMWorker);
  return xNum;
}