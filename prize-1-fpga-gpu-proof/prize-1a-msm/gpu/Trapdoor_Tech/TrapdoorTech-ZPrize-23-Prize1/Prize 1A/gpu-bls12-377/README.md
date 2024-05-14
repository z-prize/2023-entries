
## Algorithm outline and description of implemented optimizations

*  This code is based on the Matter Labs submission for the ZPRIZE 2022. And we do some integrations and optimizations. Thanks to the Matter Labs team.

*   Replace CUDA runtime API with driver API. This aims to integrate with our code.
*   Pre-allocate sort buffer with unified memory.
*   Enable Twisted Edwards to replace the  XYZZ.

## Run bench-mark：

*   Change to directory TrapdoorTech-zprize-ec-gpu-kernels.
    ```
     cd ZPrize2023-Prize1/TrapdoorTech-zprize-ec-gpu-kernels
    ```

*   Execute the following cargo command to generate the gpu-kernel.bin file.
    ```
    cargo run  --features "bls12_377, cuda" --release
    ```

*   Copy the binary file gpu-kernel-377.bin to the directory TrapdoorTech-zprize-ec-gpu-common.
    ```
    cp gpu-kernel-377.bin ../TrapdoorTech-zprize-ec-gpu-common
    ```
*   Change to directory TrapdoorTech-zprize-ec-gpu-common, then execute cargo bench
    ```
    cd ../TrapdoorTech-zprize-ec-gpu-common
    
    cargo bench
    ```