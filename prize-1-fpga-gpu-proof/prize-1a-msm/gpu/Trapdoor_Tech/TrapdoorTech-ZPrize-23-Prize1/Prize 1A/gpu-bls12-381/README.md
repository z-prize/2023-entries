
## Algorithm outline and optimizations

*  This code is based on the Matter Labs submission for the ZPRIZE 2022. And we do some integrations and optimizations. Thanks to the Matter Labs team.

*  We use 11 bits as window bits, and handle the top 2 bits alone since there are 255 bits for BLS381. And then merge the two parts as final result.
*  Replace CUDA runtime API with driver API. This aims to integrate with our code.

* Pre-allocate sort buffer with unified memory.


## Running benchmark：

*   Change to directory TrapdoorTech-zprize-ec-gpu-kernels.
    ```
     cd ./TrapdoorTech-zprize-ec-gpu-kernels
    ```

*   Execute the following cargo command to generate the gpu-kernel-381.bin file.
    ```
    cargo run  --features "bls12_381, cuda" --release
    ```

*   Copy the binary file gpu-kernel-381.bin to the directory TrapdoorTech-zprize-ec-gpu-common.
    ```
    cp gpu-kernel-381.bin ../TrapdoorTech-zprize-ec-gpu-common
    ```
*   Change to directory TrapdoorTech-zprize-ec-gpu-common, then execute cargo bench
    ```
    cd ../TrapdoorTech-zprize-ec-gpu-common/
    
    cargo bench
    ```