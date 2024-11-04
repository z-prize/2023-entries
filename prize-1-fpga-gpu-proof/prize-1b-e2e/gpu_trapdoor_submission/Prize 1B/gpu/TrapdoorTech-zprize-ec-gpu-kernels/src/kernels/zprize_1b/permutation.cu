

const uint UNIT_LEN = 4;

DEVICE fd_q::storage product(fd_q::storage array_rets[UNIT_LEN])
{
  fd_q::storage ret = array_rets[0];
  for (int i = 1; i < UNIT_LEN; i++)
  {
    ret = fd_q::mul(ret, array_rets[i]);
  }
  return ret;
}

DEVICE fd_q::storage numerator_irreducible(fd_q::storage root, fd_q::storage w, fd_q::storage k, fd_q::storage beta,
                                           fd_q::storage gamma)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;

  fd_q::storage ret = fd_q::mul(beta, k);

  ret = fd_q::mul(ret, root);
  ret = fd_q::add(w, ret);
  ret = fd_q::add(ret, gamma);

  return ret;
}

DEVICE fd_q::storage numerator_product(fd_q::storage gate_root, const fd_q::storage *gate_wires, fd_q::storage beta,
                                       fd_q::storage gamma, const fd_q::storage *ks)
{

  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;

  fd_q::storage array_rets[UNIT_LEN];
  for (int i = 0; i < UNIT_LEN; i++)
  {
    array_rets[i] =
        numerator_irreducible(gate_root, gate_wires[i], ks[i], beta, gamma);

    // printf("gid:%d, beta:%x-%x-%x-%x-%x-%x-%x-%x\n",
    //        gid, d.limbs[0], d.limbs[1], d.limbs[2], d.limbs[3], d.limbs[4], d.limbs[5], d.limbs[6], d.limbs[7]);
  }
  return product(array_rets);
}

DEVICE fd_q::storage denominator_irreducible(fd_q::storage w, fd_q::storage sigma, fd_q::storage beta,
                                             fd_q::storage gamma)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;

  fd_q::storage ret = fd_q::mul(beta, sigma);
  ret = fd_q::add(w, ret);
  ret = fd_q::add(ret, gamma);

  return ret;
}

DEVICE fd_q::storage denominator_product(const fd_q::storage *gate_sigmas, const fd_q::storage *gate_wires,
                                         fd_q::storage beta, fd_q::storage gamma)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;

  fd_q::storage array_rets[UNIT_LEN];
  for (int i = 0; i < UNIT_LEN; i++)
  {
    array_rets[i] =
        denominator_irreducible(gate_wires[i], gate_sigmas[i], beta, gamma);

    auto e = gate_sigmas[i];
  }
  return product(array_rets);
}

#define MAX_THREADS 64
#define MIN_BLOCKS 16

KERNEL void product_argument(const fd_q::storage *__restrict__ gate_roots, const fd_q::storage *__restrict__ gatewise_sigmas,
                             const fd_q::storage *__restrict__ gatewise_wires, const fd_q::storage *__restrict__ ks, const fd_q::storage *__restrict__ beta,
                             const fd_q::storage *__restrict__ gamma, fd_q::storage *__restrict__ n_dest, fd_q::storage *__restrict__ d_dest,
                             uint count)
{

  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;

  fd_q::storage gate_root = gate_roots[gid];
  const fd_q::storage *gate_wires = &gatewise_wires[gid * 4];

  n_dest[gid] = numerator_product(gate_root, gate_wires, *beta, *gamma, ks);
  const fd_q::storage *gate_sigmas = &gatewise_sigmas[gid * 4];

  d_dest[gid] = denominator_product(gate_sigmas, gate_wires, *beta, *gamma);

  d_dest[gid] = fd_q::inverse(d_dest[gid]);
  n_dest[gid] = fd_q::mul(n_dest[gid], d_dest[gid]);

}

/* pv - product value individually
 * iv - intermediate values
 * chunk_size - number of handled elements for each chunk
 * depth - depth of recursion
 */
KERNEL void product_z_part1(fd_q::storage *__restrict__ pv, fd_q::storage *__restrict__ iv, uint chunk_size, uint n)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if ((gid * chunk_size) >= n)
    return;

  uint start = gid * chunk_size;
  uint end = min(start + chunk_size, n);

  fd_q::storage acc = fd_q::get_one();

  for (uint i = start; i < end; i++)
  {
    acc = fd_q::mul(acc, pv[i]);
    pv[i] = acc;
  }

  iv[gid] = acc;
}

KERNEL void product_z_part2(fd_q::storage *__restrict__ iv, fd_q::storage *__restrict__ piv, uint chunk_size, uint n)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if ((gid * chunk_size) >= n)
    return;

  // prepare coefficients for chunks
  fd_q::storage coeff = fd_q::get_one();
  for (uint i = 0; i < gid; i++)
  {
    coeff = fd_q::mul(coeff, iv[i]);
  }

  piv[gid] = coeff;
}

KERNEL void product_z_part3(fd_q::storage *__restrict__ pv, fd_q::storage *__restrict__ zv, fd_q::storage *__restrict__ piv, uint chunk_size, uint n)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if ((gid * chunk_size) >= n)
    return;

  uint start = gid * chunk_size;
  uint end = min(start + chunk_size, n - 1);

  if (gid == 0)
  {
    zv[0] = fd_q::get_one();
  }

  // apply chunk coefficients
  for (uint i = start; i < end; i++)
  {
    zv[i + 1] = fd_q::mul(pv[i], piv[gid]);
  }
}

KERNEL void wires_to_single_gate(const fd_q::storage *__restrict__ w_l, const fd_q::storage *__restrict__ w_r, const fd_q::storage *__restrict__ w_o, const fd_q::storage *__restrict__ w_4, fd_q::storage *__restrict__ gatewise_wires, uint n)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= n)
    return;

  gatewise_wires[gid * 4] = w_l[gid];
  gatewise_wires[gid * 4 + 1] = w_r[gid];
  gatewise_wires[gid * 4 + 2] = w_o[gid];
  gatewise_wires[gid * 4 + 3] = w_4[gid];
}

#undef MAX_THREADS
#undef MIN_BLOCKS
