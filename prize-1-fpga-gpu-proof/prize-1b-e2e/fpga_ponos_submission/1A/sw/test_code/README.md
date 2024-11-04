# zprize-msm-testcase

# Introduction
 This repo is used to create MSM-tescase on curve 381 or 377.

 Command:

```cargo run --bin zprize-msm-samples --  -c 381 -d 16777216 -s 0```

- If you want to change the curve, you can change the 381 -> 377
- if you want to change the degree, you can change the data after ```-d```, like ```-d 1024```
- ```-s``` is the random seed, you can change it into different numbers to create different test cases.
