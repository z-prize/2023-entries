# PolyU Team
# Zprize: High Throughput Signature Verification

Verifying ECDSA signatures with keccak256 is not SNARK-friendly, as it involves two main challenges:
- **bitwise operations** (XOR, ROT, etc.) for the hash function.
- **non-native arithmetic** (NNA) for the elliptic curve operations that are not native to the field used by the circuit.

To address these challenges, we implement secp256k1 ECDSA signatures verification in the native field of Varuna, which includes the following building blocks,
- **16-bit lookup operations** (XOR, ROT, NOT, AND) for the keccak256 hash function based on our generalized version of [Better XOR](https://zcash.github.io/halo2/design/gadgets/sha256/table16.html).
- **non-native arithmetic** (NNA) for the secp256k1 elliptic curve operations based on [xJsnark](https://akosba.github.io/papers/xjsnark.pdf) and [Gnark's pure R1CS optimizations for non-native arithmetic](https://www.youtube.com/watch?v=05JemsgfEX4&list=PLj80z0cJm8QHm_9BdZ1BqcGbgE-BEn-3Y)
- **lookup range proof** for the hash function and non-native arithmetic.


## Performance

### Benchmark for 50 signature verification
| Msg Length (bytes) |  # of Constraints per Sig |  # of Public Var per Sig  | # of Private Var per Sig | Time for 50 sig(min) | # of Proofs for 50 sig | # of Sig per Proof
|----------|----------|----------|----------|----------|----------|----------|
| 100 | $\approx 2^{21}$| $\approx 2^{9}$ |$\approx 2^{20}$ | $\approx 15$| 2  | 25|
| 100 | $\approx 2^{21}$| $\approx 2^{9}$ |$\approx 2^{20}$ | $\approx 33$| 50  | 1|
| 1,000  | $\approx 2^{21}$ | $\approx 2^{11}$ | $\approx 2^{20}$|$\approx 16$ | 3 | 20|
| 1,000  | $\approx 2^{21}$ | $\approx 2^{11}$ | $\approx 2^{20}$|$\approx 34$ | 50  | 1|
| 50,000 | $\approx 2^{23}$ | $\approx 2^{16}$ | $\approx 2^{23}$|$\approx 155$ | 17 | 3| 
| 50,000 | $\approx 2^{23}$ | $\approx 2^{16}$ | $\approx 2^{23}$|$\approx 240$ | 50  | 1|


The benchmarks were conducted on a Mac Studio equipped with an Apple M2 Max chip and 32GB of RAM. Additionally, we configured the  mode of message, public key, and signature to be public, public, and private, respectively. You have the flexibility to customize these modes as desired in the file [API](https://github.com/MengLing-L/zprize-ecdsa-varuna/blob/main/zprize/src/api.rs#L68). 

### Remark
For a single proof, given sufficient memory capacity, the larger the number of signatures being proved, the shorter the benchmark time will be (i.e. 2 proofs x 25 sigs per proof is faster than 3 proofs x 20 sigs). Therefore, there is a trade-off between memory capacity and the number of signatures per proof. If your machine cannot support the number of signatures per proof listed in the table above, or if you have better machine than the one we used, you can modify the number of batch signatures in this file [bench](https://github.com/MengLing-L/zprize-ecdsa-varuna/blob/main/zprize/benches/bench.rs).
## Technical Overview

### Hash
Direct implementation of bitwise operations such as XOR, NOT, AND inside hash function would result in an excessively large circuit. To address this issue, we utilize lookup table on 16-bit binary string to covert XOR, NOT,AND operation into linear operations of fields. In details, we split the input into 16-bit binary strings (submessages), and the following lookup constriants will be performed on the 16-bit strings. Varuna's native field is 253-bit, and each 16-bit string will be processed as a native field element. 

In Keccak, ROT (rotate) operation is performed on 64-bit strings. In the original Varuna proof system, each message bit is considered as a variable (witness) and stored as a native field element. Thus, in the original setting, ROT operation will not increase the number of variable and there is no need to add constraints for the ROT operation. In our setting, message is dealt as a bunch of 16-bit submessages. The rotation operation will change the content of current variables and increase the number of variable. To facilitate rotation without introducing additional constraints except lookup constraints, we divide the 64 bits into two chunks: the first chunk contains $t$ bits,where $t$ is the rotation bit, and the second chunk contains the remaining 64 - $t$ bits. We further split each chunk into 16-bit subchunks. It's important to note that if $t$ is not a multiple of 16, the length of the significant bits in the last subchunk for either chunk may be less than 16. 

In the following, we first provide a detailed technical review of how we use lookup constraints to constraint XOR, NOT, and AND operations. Subsequently, we present our approach to performing the rotation operation inside the hash function.
#### XOR, NOT and AND
[Better XOR](https://zcash.github.io/halo2/design/gadgets/sha256/table16.html) provides us with a method to compute the XOR of two numbers. It defines a mapping function $f$ such that $f(x)$ pads a leading 0 to each bit of $x$. Now given 3 numbers $a$, $b$ and $c$, the proposition $a\oplus b=c$ is equivalent to:

$$\exists w, \text{s.t. } f(a)+f(b)=f(c)+2\times f(w)$$

We generalize the approach to perform the XOR operation on multiple numbers. When dealing with the keccak256 hash function, we need to perform the XOR operation consecutively on five numbers. The aforementioned method is not directly applicable in this case. Therefore, we modify $f$ such that $f(x)$ pads two leading zeros to each bit of $x$ to support the XOR operation consecutively on five numbers. Specifically, given six numbers $v, a_0, a_1, a_2, a_3$, and $a_4$, the proposition $v=\bigoplus_{i\in[0,...,4]}a_i$ is equivalent to:

$$\exists w_1,w_2,\text{s.t. } \sum_{i\in[0,...,4]}f(a_i)=f(v)+2 \times f(w_1)+4\times f(w_2)$$

In our implementation, we use one lookup table to establish the mapping between $x$ and $f(x)$. We also use another lookup table to record all the possible $f(x)$ values, which allows us to prove that some value $w'$ corresponds to a valid $f(w)$. It is worth noting that, despite the modification in the definition of $f$, the new $f$ can still be applied to the method used in [Better XOR](https://zcash.github.io/halo2/design/gadgets/sha256/table16.html). Please refer to [example of XOR operation on five numbers](https://github.com/MengLing-L/zprize-ecdsa-varuna/blob/main/zprize/src/circuit/keccak/hash.rs#L327) for the specific example and details.

As for NOT and AND operations, please refer to [NOT implementation](https://github.com/MengLing-L/zprize-ecdsa-varuna/blob/main/zprize/src/circuit/keccak/hash.rs#L497) and [AND implementation](https://github.com/MengLing-L/zprize-ecdsa-varuna/blob/main/zprize/src/circuit/keccak/hash.rs#L498) for the specific example and details.

#### ROT
In the keccak256 hash function, there are numerous bitwise operations (such as XOR), and in our code, we use the $f(x)$ values to ensure the correctness of these bitwise operations. For the ease of implementation, we convert the ROT operation on a 64-bit number $x$ in keccak256 to a ROT operation on $3\times64=192$ $f(x)$. We typically split the 192 bits into four 48-bit segments. Now for a number $x=\sum_{i\in[0,...,3]} x_i\times 2^{16i}$, we represent $f(x)$ as:

$$f(x)=\sum_{i\in[0,...,3]} f(x_i)\times 2^{48i}$$

When performing a ROT operation on $f(x)$ with a number of bits that is a multiple of 48, we only need to swap the positions of different $f(x_i)$ values. However, if the number of bits for the ROT operation is not a multiple of 48, this method is not feasible. Therefore, for $f(x)$ to be rotated by $t$ bits with $48\nmid t$,  we let $k=\lceil\frac{t}{48}\rceil$, and represent f(x) using the five values $\{f(x_i)\}_{0\leq i\leq4}$:

Now, we can simply swap the positions of different $f(x_i)$ values to represent $f(x)$ before and after the ROT operation.
It is important to note that we require:
1. $\{f(x_i)\}_{0\leq i < k}$ to exactly form the lower $t$ bits of $f(x)$
2. $\{f(x_i)\}_{k\leq i\leq4}$ to exactly form the upper $(192-t)$ bits of $f(x)$.

These conditions are enforced through range checks. Please refer to [ROT range_check](https://github.com/MengLing-L/zprize-ecdsa-varuna/blob/main/zprize/src/circuit/table/mod.rs#L51) for range check implementation and [ROT example](https://github.com/MengLing-L/zprize-ecdsa-varuna/blob/main/zprize/src/circuit/keccak/hash.rs#L343) for the specific example.

### Non-native Arithmetic
Direct implementation of non-native elliptic curve operations within the circuit field would result in an excessively large ECDSA verifier circuit. To circumvent the need for a circuit that supports non-native arithmetic, we opt to precompute and store all essential values during the execution of non-native operations. Subsequently, we validate the integrity of these values within the circuit, ensuring they are well-formed. 

In the following, we describe a method to prove the integrity of non-native multiplication and linear combinations in the circuit. Following this, we illustrate how to prove the integrity of for non-native elliptic curve operations based on non-native multiplications and linear combinations.
#### Non-native Multiplication
Consider the scenario where $a, b, c \in q_{256}$ and our objective is to demonstrate that $a \times b = c \mod q_{256}$ within the native field. It is straightforward to ascertain that $a \times b = c + \alpha \times q_{256}$ as integers. Within the native field, we can express $a, b, c, \alpha,$ and $q_{256}$ as polynomials. For example, consider $a(X) = a_1 + a_2 \times X + a_3 \times X^2$, which enables the circuit field to process $a_1, a_2, a_3$ (as well as $b, c, \alpha, q_{256}$) with bit lengths of 96, 96, and 64, respectively. Clearly, $a = a(B)$ when $B = 2^{96}$. Therefore, if following condition is met within the native field, we can ascertain that $c = a \times b \mod q_{256}$,
$$a(X) \times b(X) = c(X) + \alpha(X) \times q_{256}(X) + (B - x)e(X)\mod q_{253}$$
and 
$$a_1, b_1, c_1, \alpha_1 \in [0,\ldots, 2^{96}-1]$$
$$a_2, b_2, c_2, \alpha_2 \in [0,\ldots, 2^{96}-1]$$
$$a_3, b_3, c_3, \alpha_3 \in [0,\ldots, 2^{64}-1]$$
$$\text{coefficients of } e(X) \in [-2^{98}+1,\ldots, 2^{98}-1]$$
where $e(X) = e_2(X) - e_1(X)$. In this context, the coefficients of $e_2(X)$ represent the part of the coefficients of $f(X)$ that exceeds $B$, with $f(X) = c(X) + \alpha(X) \times q_{256}(X)$. Meanwhile, $e_1(X)$ corresponds to the values derived from $a(X) \times b(X)$. For a concrete example and further details, please refer to [non-native multiplication](https://github.com/MengLing-L/zprize-ecdsa-varuna/blob/main/zprize/src/circuit/nonnative/basic/mul.rs).
#### Non-native Linear Combination
Consider a simple case of a linear combination where $a, b, c, d \in q_{256}$ and we aim to demonstrate that $a + b - c = d \mod q_{256}$. The proof follows a similar approach as with non-native multiplication. If the condition below is met within the native field, we can validate that $d$ is indeed the result of the linear combination given by $a + b - c \mod q_{256}$:
$$a(X) + b(X) - c(X) + q_{256}(X) = d(X) + \alpha(X) \times q_{256}(X) + (B - x)e(X)\mod q_{253}$$
It is important to note that we introduce an additional polynomial $q_{256}(X)$ on the left side when subtraction is involved in the linear combination. This is because subtraction could yield negative coefficients in the resulting polynomial, which are undesirable in the native field. Moreover, the inclusion of $q_{256}(X)$ does not compromise the proof's validity.
#### Validate Non-native secp256k1 Curve Point
Given a point $P(x,y)$, we should check that $P$ indeed lies on the secp256k1 curve. To establish that a point is on the secp256k1 curve, its coordinates $(x, y)$ must satisfy the following condition:
$$x^3 + 7 = y^2 \mod q_{256}$$
and 
$$x, y \in [1,\ldots, q_{256}-1]$$
We can rewrite the constriant as:
$$x2 = x \times x \mod q_{256}$$
$$x3 = x2 \times x \mod q_{256}$$
$$y2 = y \times y \mod q_{256}$$
$$t = x3 + 7 \mod q_{256}$$
Therefore, we can precompute $x2, x3, y2, t$ over $q_{256}$ and validate these values in the circuit, ensuring they result from either non-native multiplication or a linear combination and are well-formed. If these precomputed values are well-formed, with $x, y \in \mathbb{Z}/q_{256}\mathbb{Z}$, and $y2(X) = t(X)$ holds in the native field, we can confirm that the point $P$ lies on the secp256k1 curve. Please refer to [non-native point](https://github.com/MengLing-L/zprize-ecdsa-varuna/blob/main/zprize/src/circuit/nonnative/group/mod.rs) for the specific example and details.
#### Non-native Point Addition
Given points $A(x_a, y_a), B(x_b, y_b), C(x_c, y_c)$ on the secp256k1 curve, we need to check that point $C = A + B$. If $C$ is equal to $A + B$, their coordinates must satisfy the following equations:
$$\lambda = \frac{y_a - y_b}{x_a - x_b} \mod q_{256}$$
$$x_c = \lambda^2 - x_b - x_a \mod q_{256}$$
$$y_c = \lambda \times (x_b - x_c) - y_b \mod q_{256}$$
and 
$$x_a, y_a, x_b, y_b, x_c, y_c \in [1,\ldots, q_{256}-1]$$
We can rewrite the constriant as:
$$\lambda^2 =  \lambda \times \lambda \mod q_{256}$$
$$x_{ab} = x_a - x_b \mod q_{256}$$
$$y_{ab} = y_a - y_b \mod q_{256}$$
$$x_{bc} = x_b - x_c \mod q_{256}$$
$$t_1 = \lambda \times x_{ab} \mod q_{256}$$
$$t_2 = \lambda^2 - x_b - x_a \mod q_{256}$$
$$t_3 = \lambda \times x_{bc} \mod q_{256}$$
$$t_4 = t_3 - y_b \mod q_{256}$$
Therefore, by precomputing the above values over $q_{256}$ and verifying that they are well-formed in the circuit through non-native multiplication or linear combination, we can prove the correctness of the point addition. Namely, if these precomputed values are well-formed, with $x_a, y_a, x_b, y_b \in \mathbb{Z}/q_{256}\mathbb{Z}$, $x_c, y_c$ within the range $[0, \ldots, 2^{256})$, and it holds that $x_c(X) = t_2(X)$, $y_c(X) = t_4(X)$, and $t_1(X) = y_{ab}(X)$ in the native field, then we can assert that $C = A + B$. Please refer to [non-native point addition](https://github.com/MengLing-L/zprize-ecdsa-varuna/blob/main/zprize/src/circuit/nonnative/group/add.rs) for the specific example and details.
#### Non-native Point Doubling
Given points $A(x, y)$ and $C(x_c, y_c)$ on the secp256k1 curve, we need to check that point $C = A + A$. To confirm that $C$ is equal to $A + A$, their coordinates must satisfy the following equations:
$$\lambda = \frac{3x^2}{2y} \mod q_{256}$$
$$x_c = \lambda^2 - x - x \mod q_{256}$$
$$y_c = \lambda \times (x - x_c) - y \mod q_{256}$$
and
$$x, y, x_c, y_c \in [1,\ldots, q_{256}-1]$$
The constraints can be reformulated, allowing us to check that $C = A + A$ in a manner similar to point addition. For a specific example and further details, please refer to [non-native point double](https://github.com/MengLing-L/zprize-ecdsa-varuna/blob/main/zprize/src/circuit/nonnative/group/double.rs) for the specific example and details.
#### Non-native Point Multiplication
Given points $P(x, y), C(x_c, y_c)$ on secp256k1 curve, and scalar $s\in \mathbb{Z}/p_{256}\mathbb{Z}$, we need to check that the point $C$ is correctly computed as the point multiplication $s \cdot P$. Considering the point multiplication algorithm
```
let bits = bit_representation(s) # the vector of bits (from LSB to MSB) representing scalar
let output = IDENTITY (point at infinity)
let temp = base point P
let point_add = Vec<NonNativePointAdd>
let point_double = Vec<NonNativePointDouble>
for bit in bits:
    let old_output = output
    if bit == 1:  
       output =  temp + old_output # point add
        point_add.push(NonNativePointAdd(temp, old_output, output))
    else bit == 0:
        output = IDENTITY + old_output # point add
        point_add.push(NonNativePointAdd(IDENTITY, old_output, output))

    let old_temp = temp
    temp = temp + temp # point double
    point_double.push(NonNativePointDouble(old_temp, temp))
    
return output
```
We record all the intermediate non-native point additions and doublings. If we can check that all intermediate points are correctly computed from the scalar multiplication $s \cdot P$, we can assert that $C = s \cdot P$. For a specific example and further details, please refer to [non-native point multiplication](https://github.com/MengLing-L/zprize-ecdsa-varuna/blob/main/zprize/src/circuit/nonnative/group/mul.rs) for details.
### ECDSA Signature Verification
Given the ECDSA signature $(r,s)$, messge $m$, and the public key $Q$, the signature verification algorithm first calculates,
$$R = s^{-1}\mathsf{H}(m) \cdot  P + s^{-1}r\cdot Q$$
then accepts the signature if and only if,
$$r = R.x$$
It is evident that the verification process can be decomposed into non-native multiplications and point multiplications. It is worth noting that we require an additional non-native multiplication for $s \times s^{-1} = 1 \mod p_{256}$ to ensure that $s^{-1}$ is properly calculated. Additionally, it is crucial to ensure that both $r$ and $s$ fall within the range of $[1, \ldots, p_{256}-1]$ to maintain validity.
### Range Proof
In the non-native arithmetic process, we need to prove that values are in the range $[0, 2^K - 1]$, $[-2^K + 1, 2^K - 1]$, or $[1, B - 1]$, where $K$ and $B$ can be any positive integers. Therefore, we offer three types of range proofs: two types using lookup tables and one type using bitwise comparison. For a specific example and further details, please refer to [range proof](https://github.com/MengLing-L/zprize-ecdsa-varuna/blob/main/zprize/src/circuit/helper/range_proof.rs) for details.
- **Type 1 for $[0, 2^K - 1]$:** We construct a lookup table to accommodate values ranging from 0 to $2^{16} - 1$. To prove a given value $f$ is in this range, we compute $f_1, \ldots, f_t$ such that $f = f_1 + f_2 \times 2^{16} + \ldots + f_t \times (2^{16})^{t-1}$. The value of $t$ is determined by $K/16$ if $K \% 16 = 0$, or $t = K/16 + 1$ otherwise. To ensure the validity of the range proof, we impose the lookup constraint, confirming that $f_1, \ldots, f_t$ fall within the range specified by the lookup table, namely $[0, 2^{16}-1]$.

    It is important to note that when $K \% 16 \neq 0$, the bit length of $f_t$ is $d = K \% 16$. We establish that $f$ lies within the range $[0, 2^K - 1]$ precisely when $f_t$ is within the range $[0, 2^d - 1]$. To satisfy this requirement, we enforce the lookup constraint to verify that $f_t \times 2^{16-t}$ also falls within the lookup table range.
- **Type 2 for $[-2^K + 1, 2^K - 1]$:** We construct a lookup table for values ranging from $-2^K + 1$ to $2^{K} - 1$. If the value being proven, denoted as $f$, falls within the range $[0, 2^K - 1]$, the proving method is the same as in the first type. However, when given a value $f$ within the range $[-2^K + 1, 0)$, we follow a slightly different approach.
  
    First, we compute $f' = q_{253} - f$, where $q_{253}$ is the native modulus. Next, we calculate $f'_1, \ldots, f'_t$ for $f'$ using a similar methodology as in the first type. This allows us to express $f$ as the sum $(-f'_1) + (-f'_2) \times 2^{16} + \ldots + (-f'_t) \times (2^{16})^{t-1}$. To ensure the validity of the proof, we enforce the lookup constraint, verifying that $-f'_1, \ldots, -f'_t$ fall within the lookup table range. Additionally, when $K \% 16 \neq 0$, we guarantee that $(-f'_t) \times 2^{16-t}$ also lies within the lookup table range.

    By following this approach, we can effectively demonstrate that $f$ falls within the range $[-2^K + 1, 2^K - 1]$.

- **Type 3 for $[1, B - 1]$:** We begin by converting both the proving value and the bound $B$, into their respective bit representations. We then compare these two values from the least significant bit upwards to the most significant bits to ensure that $f$ is smaller than the bound $B$ (The  bit comparison has been implemented in snarkVM).

    In our non-native arithmetic, the non-native value is represented using three native field elements $f_1, f_2, f_3$, each with lengths of 96, 96, and 64 bits, respectively. When converting back the proving value into its 256-bit representation, we extract the lowest 96, 96, and 64 bits from the three native field elements. Additionally, we verify that the highest 157, 157, and 198 bits of the three native field elements, respectively, are all zero.

    Furthermore, to prove that the proving value falls within the range of $[1, B-1]$, it is essential to demonstrate that $f$ is not equal to zero. By ensuring that the highest 157, 157, and 198 bits of the three native fields are zero, we can assert that if the sum $f_1 + f_2 + f_3$ is not equal to zero, then $f$ is also not equal to zero.

    By following these steps, we establish the validity of the proving value and ensure that it lies within the desired range of $[1, B-1]$.


## Remark
We fork the zprize_2023 branch of the snarkVM repository and make modifications to the assignment in snarkVM to enable the enforcement of the lookup constraint in the Varuna proof system. Throughout this process, we treat Varuna as a black box and do not modify the protocol of Varuna.
