This project utilizes GPU acceleration to enhance the proving performance for the Poseidon-Merkel tree circuit. All relevant GPU-accelerated code can be found within the `./gpu` directory of this repository. As of now, MSM/FFT computations has been transitioned to utilize GPU acceleration. However, it's important to note that this implementation currently serves as a framework. Many computations within the project still rely on CPU processing rather than taking full advantage of GPU capabilities.

## How to run?
```
export PATH=$PATH:/usr/local/cuda/bin/
cargo run
```

## How to generate/update kernel image?
```
cd ./gpu/TrapdoorTech-zprize-ec-gpu-kernels
cargo run --release
cp gpu-kernel.bin ../../merkle-tree
```

Wishing you all an enjoyable development experience.
