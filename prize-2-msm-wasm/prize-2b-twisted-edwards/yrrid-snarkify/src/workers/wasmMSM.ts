import init, * as Aleo from './aleo-wasm-web';
import { expose } from 'threads/worker';

await init();
await Aleo.initThreadPool(Math.max(1, navigator.hardwareConcurrency - 2));

export const wasmMSM = async (groups: string[], scalars: string[]): Promise<string> => {
  const xresult = Aleo.Address.msm(groups, scalars);
  const xNum = xresult.slice(0, xresult.indexOf('group'));
  return xNum;
}

expose(wasmMSM);