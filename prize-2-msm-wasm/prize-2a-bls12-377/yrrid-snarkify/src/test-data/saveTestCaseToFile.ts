export const savePointsToFile = async (points: { x: string, y: string, z: string }[]) => {
    const data = points.map(point => `{ "x": "${point.x}", "y": "${point.y}", "z": "${point.z}"}`).join('\n');
    const blob = new Blob([data], { type: 'text;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'points');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
};

export const saveScalarsToFile = async (scalars: bigint[]) => {
    const data = scalars.map(scalar => `"${scalar.toString()}",`).join('\n');
    const blob = new Blob([data], { type: 'text;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'scalars');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
};
