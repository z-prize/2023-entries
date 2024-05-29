## ZPrize 2023 Submissions ##

Welcome to the repository of ZPrize 2023 submissions. Below you will find information about the different prize categories, a summary of the competition results, and links to additional information about the ZPrize competition in general.

### Prize Categories ### 

The ZPrize 2023 competition featured the following prize categories:

####  Prize 1: Accelerating ZKPs using Specialized Hardware (GPU/FPGA) #### 

The goal of Prize 1 was to accelerate ZKP proving in both GPUs and FPGA-type hardware. This prize was split into two subcategories: one focused on multiscalar multiplication, and the second on end-end proof-generation.

**Prize 1a - Accelerating MSM on GPU/FPGA**: This prize was a sequel to the 2022 competition, which focused solely on accelerating multiscalar multiplication, defined as:

**Prize 1b - Accelerating E2E Proof Generation**: This subcategory focuses on taking the result from Prize 1a and integrating it into a full end-to-end proof using the Varuna proof system (a variant of universal proof system Marlin that incorporates lookups & other
improvements)

#### Prize 2: Accelerating Multi-scalar multiplication for web clients  using WebGPU/WASM ####

The focus of Prize 2 was optimizing MSM operations using web-based technologies, such as WebGPU and WASM, to enhance client-side proving performance on consumer-grade devices. Competitors were required to compute an MSM over either the BLS 12-377 curve or the Twisted Edwards Curve. In addition, we had a subcategory specifically for implementations that used WebGPU.

**Prize 2a - BLS12-377**: WASM MSM improvements for the BLS12-377 curve

**Prize 2b - Twisted Edwards**: WASM MSM improvements for the Twisted Edwards curve

**WebGPU Specific Implementations**: Subcategory specifically devoted to submissions tha used WebGPU 

### Prize 3: ECDSA Signature Verification in Zero-Knowledge Proof Systems ### 

The focus of Prize 3 was accelerating a real-world use case: batch non-native signature verification inside of a Varuna SNARK (a variation of Marlin that includes lookups)

## Results ##

### Prize 1 ### 

#### Prize 1a (GPU) ####

| Position         | Team                            | BLS 12-377 | BLS 12-381 | Avg    | Link |
|------------------|---------------------------------|------------|------------|--------|------|
| 1                | Marco Zhou of StorSwift         | 430 ms     | 525 ms     | 478 ms |      |
| 2                | Trapdoor Tech                   | 598 ms     | 641 ms     | 620 ms |      |
| Special Mention* | Yrrid Software and Snarkify ZKP | 367 ms     | 440 ms     | 404 ms |      |
| Baseline         | N/A                             | 472 ms     |            |        |      |

Note: Yrrid Software submission was not counted in the competition due to late submission

#### Prize 1a (FPGA) #### 

| Position | Team                                    | BLS 12-377        | BLS 12-381        | Avg     | Link |
|----------|-----------------------------------------|-------------------|-------------------|---------|------|
| 1        | Yrrid Software (Niall Emmart & Tony Wu) | 2020 ms (250 MHz) | 2024 ms (250 MHz) | 2022 ms |      |
| 2        | Superscalar                             | 1431 ms (280 MHz) | N/A               | 2146 ms |      |
| 3        | Ponos                                   | 2906 ms (250 MHz) | 3904 ms (205 MHz) | 3405 ms |      |
| 4        | MIT / Georgia Tech                      | N/A               | N/A               | N/A     |      |
| Baseline | N/A                                     | 1465 ms           |                   |         |      |

#### Prize 1b ####

Competition ongoing. Stay tuned!

<br>

## Prize 2 ## 

### Prize 2a (BLS12-377)

| Position | Team                      | Time (ms) | Link |
|----------|---------------------------|-----------|------|
| 1        | Yrrid Software & Snarkify | 939       |      |
| 2        | Gregor Mitscha-Baude      | 2263      |      |
| Baseline | N/A                       | 143735    |      |


### Prize 2b (Twisted Edwards) ### 

| Position | Team                      | Time (ms) | Link |
|----------|---------------------------|-----------|------|
| 1        | Yrrid Software & Snarkify | 500       |      |
| 2        | Gregor Mitscha-Baude      | 1613      |      |
| 3        | Storswift                 | 3364      |      |
| 4        | Chengyuan                 | 2195      |      |
| Baseline | N/A                       | 68220     |      |


### WebGPU ### 

| Position | Team                                 | Time  | Link |
|----------|--------------------------------------|-------|------|
|          | Tal Derei (Penumbra) and Koh Wei Jie | 1451  |      |
|          | Storswift                            | 3364  |      |
| Baseline | N/A                                  | 68220 |      |

<br>

## Prize 3 ## 

| Position | Team           | Time (min) | Link |
|----------|----------------|------------|------|
| 1        | Zproof         | 10         |      |
| 1        | HK Polytechnic | 10         |      |
| Baseline | N/A            |            |      |

For a full rundown of Prize 3, check out architect ZKSecurity's [blog post](https://www.zksecurity.xyz/blog/posts/zprize-final/)

## License Information ## 

The contributions in this repository are and will always be an open-source public good.
All contributions are dual-licensed under the MIT and Apache 2.0 licenses. This means you
can choose either license to use the code in this repository.

MIT License: A permissive license that is short and to the point. It lets you do almost
anything you want with this code, as long as you include the original copyright and
license notice in any substantial portion of the code.  

Apache License 2.0: A permissive license similar to the MIT License, but also provides an
express grant of patent rights from contributors to users.

For more information, see the LICENSE.md file

