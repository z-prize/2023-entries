import { BigIntPoint } from "../reference/types";

export type TestCase = {
  baseAffinePoints: BigIntPoint[];
  scalars: bigint[];
  expectedResult: { x: bigint, y: bigint };
};

export type PowersTestCase = 16 | 17 | 18 | 19 | 20;

const getExpectedResult = (testCase: PowersTestCase): { x: bigint, y: bigint } => {
  switch (testCase) {
    case 16:
      return { x: BigInt('4490298471131273381350715833932091894064554978284853693957586604825823442429'), y: BigInt('207233051598812890797414182362695316831408959017076683749810755208551572458') };
      break;
    case 17:
      return { x: BigInt('405755281347735151880827575059343698498813029460786026451708154294960743560'), y: BigInt('7112985356832152643523650125935205310677117771129806490701829425450717492869')}
      break;
    case 18:
      return { x: BigInt('4020134989704514076121556080357844499902614818105934254331815581426895427831'), y: BigInt('2694327822589008080344499645494473764166611881342421427746308662023437975766')}
      break;
    case 19:
      return { x: BigInt('3856727778963570638772781884183843350150969534777451295534564482755471873113'), y: BigInt('1398750101296346671684024297455637342909036274728274942667983346895370713922')}
      break;
    case 20:
      return { x: BigInt('5201851187583570844529445080011852189038251929148722905178398320328749074909'), y: BigInt('3586360219804356686204324370397321114669962278596135149389460948678051407803')}
      break;
    default:
      return { x: BigInt(0), y: BigInt(0) };
      break;
  }
};

export const loadTestCase = async (testCase: PowersTestCase): Promise<TestCase> => {
  const pointsFile = await fetch(`test-data/points/${testCase}-power-points.txt`);
  const pointsText = await pointsFile.text();
  const pointsLines = pointsText.trim().split('\n');
  const points: BigIntPoint[] = pointsLines.map(line => JSON.parse(line, (key, value) => {
    if (typeof value === 'string') {
      return BigInt(value.slice());
    }
    return value;
  }));
  const scalarsFile = await fetch(`test-data/scalars/${testCase}-power-scalars.txt`);
  const scalarsText = await scalarsFile.text();
  const scalarsLines = scalarsText.trim().split('\n');
  const scalars: bigint[] = scalarsLines.map(line => BigInt(line));

  const expectedResult = getExpectedResult(testCase);

  return { baseAffinePoints: points, scalars, expectedResult };
};

