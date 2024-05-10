import * as Aleo from '@demox-labs/gpu-wasm-expose';
import { savePointsToFile } from './saveTestCaseToFile';

// Function to generate a random number
function getRandomNumber(): number {
    return Math.floor(Math.random() * 10000000000000); // Adjust the range as needed
}

// Function to create a random affine point
function createRandomAffinePoint(): { x: string, y: string } {
    return {
        x: getRandomNumber().toString(),
        y: getRandomNumber().toString()
    };
}

// Function to generate and save points
export const generateAndSaveAffinePoints = (numberOfPoints: bigint) => {
    console.log(numberOfPoints);
    const points: { x: string, y: string, z: string }[] = [];
    for (let i = 0n; i < numberOfPoints; i++) {
        
        const affineResult = Aleo.Address.random_g1_point();
        if (i % 10000n === 0n) {
            console.log(`${i} points generated.`);
        }
        const xresult = affineResult.slice(affineResult.indexOf('x=') + 2, affineResult.indexOf(', y='));
        const yresult = affineResult.slice(affineResult.indexOf('y=') + 2, affineResult.indexOf(')'));
        points.push({ x: xresult, y: yresult, z: '1'});
    }
    savePointsToFile(points);
}