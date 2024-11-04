/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

__device__ __forceinline__ void write_scalar(bls12381_fr* out, const bls12381_fr& value) {
  uint32_t lsb=threadIdx.x & 0x01;

  // pairs of threads work together to write a each value
  ((uint4*)out)[lsb]=(lsb==0) ? value.get_low() : value.get_high();
}

__global__ void witness_kernel(witness_t witness, bls12381_fr* blinding, bls12381_fr* poseidon, bls12381_fr* nodes, uint32_t count) {
  uint32_t               node_index=(blockIdx.x*blockDim.x+threadIdx.x)>>1, write_index=node_index*193 + 4;
  bls12381_fr            convert, state[3], next_state[3];    
  bls12381_fr*           pi_scalar=witness.pi_scalar;
  bls12381_fr*           wsl=witness.w_scalar_l;
  bls12381_fr*           wsr=witness.w_scalar_r;
  bls12381_fr*           ws4=witness.w_scalar_4;
  bls12381_fr*           wso=witness.w_scalar_o;
  __shared__ bls12381_fr params[209];

  for(int i=threadIdx.x;i<209;i++) {
    convert=bls12381_fr::zero();
    if(i<8)
      convert=blinding[i];
    else if(i<206)
      convert=poseidon[i-8];
    params[i]=bls12381_fr::to_montgomery(convert);
  }

  __syncthreads();

  if(node_index>=count)
    return;

  if(node_index==0) {
    write_scalar(&wsl[0], bls12381_fr::zero());
    write_scalar(&wsl[1], params[0]);
    write_scalar(&wsl[2], params[4]);
    write_scalar(&wsl[3], params[4]);

    write_scalar(&wsr[0], bls12381_fr::zero());
    write_scalar(&wsr[1], params[1]);
    write_scalar(&wsr[2], params[5]);
    write_scalar(&wsr[3], params[5]);
   
    write_scalar(&ws4[0], bls12381_fr::zero());
    write_scalar(&ws4[1], params[2]);
    write_scalar(&ws4[2], params[6]);
    write_scalar(&ws4[3], bls12381_fr::zero());

    write_scalar(&wso[0], bls12381_fr::zero());
    write_scalar(&wso[1], params[3]);
    write_scalar(&wso[2], params[7]);
    write_scalar(&wso[3], bls12381_fr::zero());
  }

  next_state[0]=bls12381_fr::r() + bls12381_fr::r() + bls12381_fr::r();
  next_state[1]=bls12381_fr::to_montgomery(bls12381_fr::load(&nodes[node_index*3+0]));
  next_state[2]=bls12381_fr::to_montgomery(bls12381_fr::load(&nodes[node_index*3+1]));

  state[0]=next_state[0] + params[17];
  state[1]=next_state[1] + params[18];
  state[2]=next_state[2] + params[19];

  write_scalar(&wsl[write_index+0], next_state[0]);
  write_scalar(&wsl[write_index+1], next_state[1]);
  write_scalar(&wsl[write_index+2], next_state[2]);

  write_scalar(&wsr[write_index+0], bls12381_fr::zero());
  write_scalar(&wsr[write_index+1], bls12381_fr::zero());
  write_scalar(&wsr[write_index+2], bls12381_fr::zero());

  write_scalar(&ws4[write_index+0], bls12381_fr::zero());
  write_scalar(&ws4[write_index+1], bls12381_fr::zero());
  write_scalar(&ws4[write_index+2], bls12381_fr::zero());

  write_scalar(&wso[write_index+0], state[0]);
  write_scalar(&wso[write_index+1], state[1]);
  write_scalar(&wso[write_index+2], state[2]);

  for(int32_t r=0;r<63;r++) {
    write_scalar(&wsl[write_index+r*3+3], state[0]);
    write_scalar(&wsl[write_index+r*3+4], state[0]);
    write_scalar(&wsl[write_index+r*3+5], state[0]);

    write_scalar(&wsr[write_index+r*3+3], state[1]);
    write_scalar(&wsr[write_index+r*3+4], state[1]);
    write_scalar(&wsr[write_index+r*3+5], state[1]);

    write_scalar(&ws4[write_index+r*3+3], state[2]);
    write_scalar(&ws4[write_index+r*3+4], state[2]);
    write_scalar(&ws4[write_index+r*3+5], state[2]);

    state[0]=state[0].sqr().sqr() * state[0];
    if(r<4 || r>=59) {
      state[1]=state[1].sqr().sqr() * state[1];
      state[2]=state[2].sqr().sqr() * state[2];
    }
    next_state[0]=state[0] * params[8] + state[1] * params[9] + state[2] * params[10];
    next_state[1]=state[0] * params[11] + state[1] * params[12] + state[2] * params[13];
    next_state[2]=state[0] * params[14] + state[1] * params[15] + state[2] * params[16];
    state[0]=next_state[0] + params[r*3 + 20];
    state[1]=next_state[1] + params[r*3 + 21];
    state[2]=next_state[2] + params[r*3 + 22];

    write_scalar(&wso[write_index+r*3+3], state[0]);
    write_scalar(&wso[write_index+r*3+4], state[1]);
    write_scalar(&wso[write_index+r*3+5], state[2]);
  }

  write_scalar(&wsl[write_index+192], state[1]);
  write_scalar(&wsr[write_index+192], state[1]);
  write_scalar(&ws4[write_index+192], bls12381_fr::zero());
  write_scalar(&wso[write_index+192], bls12381_fr::zero());

  if(node_index==count-1) {
    write_scalar(&pi_scalar[write_index+193], -state[1]);

    write_scalar(&wsl[write_index+193], state[1]);
    write_scalar(&wsr[write_index+193], bls12381_fr::zero());
    write_scalar(&ws4[write_index+193], bls12381_fr::zero());
    write_scalar(&wso[write_index+193], bls12381_fr::zero());
  }
}

merkle_node_t* merkle_tree(const st_t* non_leaf_nodes, const st_t* leaf_nodes) {
  uint32_t       height, node_count, interior_count, leaf_count, level_size, from, to, current;
  st_t*          hashes;
  merkle_node_t* nodes;

  height=MERKLE_HEIGHT;
  node_count=MERKLE_NODE_COUNT;
  leaf_count=N_LEAF_NODES;
  interior_count=N_NON_LEAF_NODES;

  hashes=(st_t*)malloc(sizeof(st_t)*node_count);
  for(int i=0;i<interior_count;i++)
    hashes[i]=non_leaf_nodes[i];
  for(int i=0;i<leaf_count;i++)
    hashes[i+interior_count]=leaf_nodes[i];

  nodes=(merkle_node_t*)malloc(sizeof(merkle_node_t)*interior_count);

  // reverse the tree, so the hashes are in the same order as the witness table
  from=0;
  current=interior_count;
  for(int level=0;level<height-1;level++) {
    level_size=1<<level;
    current=current-level_size;
    to=current;
    for(int i=0;i<level_size;i++) {
      nodes[to].left=hashes[from*2+1];
      nodes[to].right=hashes[from*2+2];
      nodes[to++].hash=hashes[from++];
    }
  }
  free(hashes);
  return nodes;
}

void store(const char* path, st_t* data, uint32_t count) {
  FILE* file=fopen(path, "w");

  fprintf(stderr, "Writing %s\n", path);
  for(uint32_t i=0;i<count;i++)
    fprintf(file, "%016lX%016lX%016lX%016lX\n", data[i].l[3], data[i].l[2], data[i].l[1], data[i].l[0]);
  fclose(file);
}

void round0(Context* context, const uint64_t* blinding_factors, const uint64_t* non_leaf_nodes, const uint64_t* leaf_nodes) {
  merkle_node_t* nodes;
  uint32_t       interior_node_count=N_NON_LEAF_NODES;

  $CUDA(cudaMemcpy(context->bf, blinding_factors, 32ul*8, cudaMemcpyHostToDevice)); 

  nodes=merkle_tree((const st_t*)non_leaf_nodes, (const st_t*)leaf_nodes);
  $CUDA(cudaMemcpy(context->tree, nodes, sizeof(merkle_node_t)*interior_node_count, cudaMemcpyHostToDevice));
  free(nodes);  

  witness_kernel<<<(interior_node_count+127)/128, 256>>>(context->w, context->bf, context->pp, context->tree, interior_node_count);
}
