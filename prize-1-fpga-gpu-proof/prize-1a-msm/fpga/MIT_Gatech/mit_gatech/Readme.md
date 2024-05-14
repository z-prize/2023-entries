# Overview 
In this repo, we design the HLS-based multi-scalar multiplication implementation and deploy it on the Alveo U250 through AXI interface.

# Step-by-Step Instructions
1. Download the Zprize official test library
```
git clone https://github.com/cysic-labs/ZPrize-23-Prize1
```

2. Generate Test Vector of ZPrize-23-Prize1
```
cd ZPrize-23-Prize1/Prize\ 1A/test_code
cargo run --bin zprize-msm-samples --  -c 381 -d 2 -s 0
```

where `-d` is to change the degree, and `-c` controls the curve with possible value as 377, 381, for BLS12-371, and BLS12-381 separately.

3. Compile the msm.c program.
```
cd mit_gatech
mkdir work
cd work
cmake ..
make -j
./msm <path_to_points.csv> <path_to_scalar.csv>
```
