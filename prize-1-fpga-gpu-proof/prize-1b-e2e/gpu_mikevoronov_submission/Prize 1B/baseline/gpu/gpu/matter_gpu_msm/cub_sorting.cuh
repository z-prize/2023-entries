#pragma once
#include <cub/cub.cuh>

static cudaError_t sort_pairs(void *d_temp_storage, size_t *temp_storage_bytes, const unsigned *d_keys_in, unsigned *d_keys_out, const unsigned *d_values_in,
                              unsigned *d_values_out, int num_items, int begin_bit, int end_bit, cudaStream_t stream)
{
  size_t ret_temp_storage_bytes = 0;
  cudaError_t ret;

  // todo stream
  if (*temp_storage_bytes == 0)
  {
    ret = cub::DeviceRadixSort::SortPairs(d_temp_storage, ret_temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, begin_bit, end_bit, stream);
    *temp_storage_bytes = ret_temp_storage_bytes;
  }
  else
  {
    ret_temp_storage_bytes = *temp_storage_bytes;

    ret = cub::DeviceRadixSort::SortPairs(d_temp_storage, ret_temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, begin_bit, end_bit, stream);
  }

  return ret;
}

static cudaError_t sort_pairs_descending(void *d_temp_storage, size_t *temp_storage_bytes, const unsigned *d_keys_in, unsigned *d_keys_out,
                                         const unsigned *d_values_in, unsigned *d_values_out, int num_items, int begin_bit, int end_bit, cudaStream_t stream)
{
  cudaError_t ret;

  if (*temp_storage_bytes == 0)
  {
    size_t ret_temp_storage_bytes = 0;
    ret = cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, ret_temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, begin_bit, end_bit, stream);
    *temp_storage_bytes = ret_temp_storage_bytes;
  }
  else
  {
    size_t ret_temp_storage_bytes = *temp_storage_bytes;
    ret = cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, ret_temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, begin_bit, end_bit, stream);
  }

  return ret;
}

static cudaError_t run_length_encode(void *d_temp_storage, size_t *temp_storage_bytes, const unsigned *d_in, unsigned *d_unique_out, unsigned *d_counts_out,
                                     unsigned *d_num_runs_out, int num_items, cudaStream_t stream)
{
  cudaError_t ret;
  if (*temp_storage_bytes == 0)
  {
    size_t ret_temp_storage_bytes = 0;
    ret = cub::DeviceRunLengthEncode::Encode(d_temp_storage, ret_temp_storage_bytes, d_in, d_unique_out, d_counts_out, d_num_runs_out, num_items, stream);
    *temp_storage_bytes = ret_temp_storage_bytes;
  }
  else
  {
    size_t ret_temp_storage_bytes = *temp_storage_bytes;
    ret = cub::DeviceRunLengthEncode::Encode(d_temp_storage, ret_temp_storage_bytes, d_in, d_unique_out, d_counts_out, d_num_runs_out, num_items, stream);
  }

  return ret;
}

static cudaError_t exclusive_sum(void *d_temp_storage, size_t *temp_storage_bytes, const unsigned *d_in, unsigned *d_out, int num_items, cudaStream_t stream)
{

  cudaError_t ret;
  if (*temp_storage_bytes == 0)
  {
    size_t ret_temp_storage_bytes = 0;
    ret = cub::DeviceScan::ExclusiveSum(d_temp_storage, ret_temp_storage_bytes, d_in, d_out, num_items, stream);
    *temp_storage_bytes = ret_temp_storage_bytes;
  }
  else
  {
    size_t ret_temp_storage_bytes = *temp_storage_bytes;
    ret = cub::DeviceScan::ExclusiveSum(d_temp_storage, ret_temp_storage_bytes, d_in, d_out, num_items, stream);
  }

  return ret;
}