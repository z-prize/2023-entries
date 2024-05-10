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
      return { x: BigInt('94006842082116618334698674554269938560504658220442275405704974851793018623976750030932275315377339755327327987799'), y: BigInt('20373698276638985490622302772174938574967913528479846848006540077491753947648956036093654307050792702539840457541') };
      break;
    case 17:
      return { x: BigInt('206224560584082546776307678440614275320062113355561962308721799926405988566792861311857124914191508657092244026797'), y: BigInt('211505771810605149801236229583532591257930087722075039263647957125630724803810862016000585191202320499088754389346')}
      break;
    case 18:
      return { x: BigInt('213590253091531711003295174396041900486736230199904022674226470027355022490783453188751023812621283421365133044335'), y: BigInt('166168294849747437548140695864136486986897221068029518430368940173172785864820517559403857089626657281214248033436')}
      break;
    case 19:
      return { x: BigInt('227918075012010659569854027573177112762469117095506192259456355647196733855535622181356473956903755312919537388289'), y: BigInt('232048820726736272000228087347068589163288439026577981179126188061989792518064409423298246183820422050991578154066')}
      break;
    case 20:
      return { x: BigInt('105645455159295492078411402285457085811978509815703136952786959329738979428758249440990135440135199333488003965024'), y: BigInt('217434031274260429359512002379640961971443333898312105830518865556255108267359047513395163712830071551228264849716')}
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

