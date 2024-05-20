/**
 * Copyright (C) 2019-2021 Xilinx, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#include <omp.h>
#include <vector>
#include <unistd.h>
#include <cmath>

#include "xcl2.hpp"
#include "field/field.h"
#include "curve/bls12_377/bls12_377_config.h"
#include "msm/msm.h"
#include "curve/twisted_edwards_extended/edwards.h"
#include "curve/curve.h"

using namespace msm_cmodel;
using namespace CURVE_BLS12_377;

#define LOAD_FROM_RUST 1
#define DEBUG_INFO 0
#define PROFILING_INFO 0

#if PROFILING_INFO
    #define START_TIMER(start) gettimeofday(&start, NULL);

    #define END_TIMER(end, start, result, message) \
        gettimeofday(&end, NULL); \
        result = (end.tv_sec * TIMECONVERTER + end.tv_usec) - (start.tv_sec * TIMECONVERTER + start.tv_usec); \
        printf("      [HOST][%s] %s elapsed time:   %ld us\n", __FUNCTION__, message, result);
#else
    #define START_TIMER(start)
    #define END_TIMER(end, start, result, message)
#endif

#define DATA_SIZE 256
#define SW_AFFINE_BYTES 96
#define SW_PROJECTIVE_BYTES 144
#define EXED_AFFINE_BYTES 144
#define EXED_PROJECTIVE_BYTES 192
#define FR_BYTES 32
#define GROUP_BATCHES_NUM_MAX 22
#define POINTS_NUM_MAX 16777216 // 2^24

#define SLICES (BIT_S * 3)
#define S_BIT2  (BIT_S * 2)
#define S_MASK ((uint64_t)(1 << BIT_S) - 1)
#define L2 ((uint64_t)(1 << (BIT_S-1)))

#define DWORD 64

#define RIGHT_SHIFT_1   (SLICES * 1 % DWORD)        //48*1%64 = 48
#define LEFT_SHIFT_1    (DWORD - RIGHT_SHIFT_1)     //16

#define RIGHT_SHIFT_2   (SLICES * 2 % DWORD)        //96%64 = 32
#define LEFT_SHIFT_2    (DWORD - RIGHT_SHIFT_2)     //32

#define RIGHT_SHIFT_3   (SLICES * 3 % DWORD)        // 3*48%64 = 16
#define LEFT_SHIFT_3    (DWORD - RIGHT_SHIFT_3)     // 48

#define RIGHT_SHIFT_4   (SLICES * 4 % DWORD)        //0
#define LEFT_SHIFT_4    (DWORD - RIGHT_SHIFT_4)     //64

#define RIGHT_SHIFT_5   (SLICES * 5 % DWORD)        //48
#define LEFT_SHIFT_5    (DWORD - RIGHT_SHIFT_5)     //16

char carry_array[POINTS_NUM_MAX]={0};

struct scalarStruct				//256bit 
{
    uint64_t dword[4];			//dword is 64bit
};

pthread_mutex_t mtx1 = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t signal1 = PTHREAD_COND_INITIALIZER;
pthread_mutex_t mtx2 = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t signal2 = PTHREAD_COND_INITIALIZER;
pthread_cond_t signal3 = PTHREAD_COND_INITIALIZER;
pthread_cond_t signal4 = PTHREAD_COND_INITIALIZER;

int post_task_flag[GROUP_BATCHES_NUM_MAX] = {0};
int pre_task_flag[GROUP_BATCHES_NUM_MAX] = {0};

struct PreThreadData
{
    uint64_t *all_slices;
    char *input_scalars;
    unsigned msm_size;
    unsigned group_batches_num;      // the number of group batches for all msm bathes.
    unsigned groups_num_each_batch;
    unsigned bit_s;
    unsigned *slice_bit;
};

int load_data(char *data, char *filepath, uint64_t element_size, uint64_t count, int mode)
{
    std::ifstream file;

    file.open(filepath, std::ios::in | std::ios::binary);
    if (!file.is_open())
    {
        printf("Error: open %s failed!\n", filepath);
        return -1;
    }

    file.seekg(0, std::ios::end);
    long long file_size = file.tellg();

    if (mode == 1)
    {
        file.seekg(8, std::ios::beg);
    }
    else
    {
        file.seekg(0, std::ios::beg);
    }

    long long file_count = file_size / element_size;

    if (count > file_count)
    {
        printf("Error: count: %ld out of file_count: %ld\n", count, file_count);
        return -1;
    }
    else
    {
        printf("OK: load count: %ld, file count: %ld\n", count, file_count);
    }

    long long need_file_size = count * element_size;
    if (need_file_size >= (2LL * 1024 * 1024 * 1024)) // over 2GB
    {
        long long file_size_offset = 0; 
        while(file_size_offset < need_file_size)
        {
            long long remaining_bytes = need_file_size - file_size_offset;
            long long bytes_to_read = remaining_bytes < (1LL * 1024 * 1024 * 1024) ? remaining_bytes : (1LL * 1024 * 1024 * 1024);
            file.read(data + file_size_offset, bytes_to_read);
            file_size_offset += bytes_to_read;
        }
    }
    else
    {
        // printf("    count: %ld, sizeof(T): %ld, read size: %ld\n", count, sizeof(T), count * sizeof(T));
        file.read(data, count * element_size);
    }


    file.close();

    return 0;
}

inline uint64_t signed_slices(uint64_t slices, char *carry_point, int mult_msm_flag)
{
    register uint64_t slice,slice0,slice1,slice2,slice_true,slice_false;
    register char flag, carry;

    carry = *carry_point;

    //slice0
    slice = slices & S_MASK;

    slice += carry;
    flag = slice > L2;
    slice_true = (1 << BIT_S) - slice;
    slice_true = (slice_true-1);
    slice_true = slice_true | (1<<(BIT_S-1));
    slice_false = slice - 1;
    slice0 = flag ? slice_true : slice_false;
    carry  = (mult_msm_flag == 1) ? 0 : flag;

    //slice1
    slice = slices >> BIT_S;
    slice = slice & S_MASK;

    slice += carry;
    flag = slice > L2;
    slice_true = (1 << BIT_S) - slice;
    slice_true = (slice_true-1);
    slice_true = slice_true | (1<<(BIT_S-1));
    slice_false = slice - 1;
    slice1 = flag ? slice_true : slice_false;
    carry  = (mult_msm_flag == 2) ? 0 : flag;

    //slice2
    slice = slices >> S_BIT2;
    slice = slice & S_MASK;

    slice += carry;
    flag = slice > L2;
    slice_true = (1 << BIT_S) - slice;
    slice_true = (slice_true-1);
    slice_true = slice_true | (1<<(BIT_S-1));
    slice_false = slice - 1;
    slice2 = flag ? slice_true : slice_false;
    carry  = (mult_msm_flag == 3) ? 0 : flag;

    //combine slices
    slice0 = slice0 & S_MASK;
    slice1 = (slice1 & S_MASK) << (BIT_S*2);
    slice2 = (slice2 & S_MASK) << (BIT_S*3);

    *carry_point = carry;

    return slice0 | slice1 | slice2;
}

void *pre_worker_thread(void *arg)
{
    printf("      [HOST][%s] launch pthread\n", __FUNCTION__);
    PreThreadData *data = static_cast<PreThreadData *>(arg);
    scalarStruct *input_scalars = (scalarStruct *)data->input_scalars;
    unsigned long time = 0;
    struct timeval time_start, time_end;

    vector<unsigned> index_start;
    vector<unsigned> index_end;
    vector<unsigned> m;
    vector<unsigned> n;
    vector<unsigned> m_shift;
    vector<unsigned> n_shift;

    get_slices_info_64bit(data->groups_num_each_batch, data->bit_s, index_start, index_end, m, n, m_shift, n_shift);

    std::vector<uint64_t*> slice_arrays;

    for (int i = 0; i < data->group_batches_num; i++)
    {
        uint64_t *slice_array = data->all_slices + i * data->msm_size;
        slice_arrays.push_back(slice_array);
    }
    omp_set_num_threads(3);

    uint64_t *slice_ptr = nullptr;
    START_TIMER(time_start);
    slice_ptr = slice_arrays[0];
	// calc slice0/1/2
    #pragma omp parallel for
    for(int i = 0; i < data->msm_size; i++)
    {
        register uint64_t t1,t2;
        t1 = input_scalars[i].dword[0];

        slice_ptr[i] = signed_slices(t1, &carry_array[i], 0);
	}
    END_TIMER(time_end, time_start, time, "get slices batch0 0/1/2 cpu");

    // Send signal1 after first group batch.
    pthread_mutex_lock(&mtx1);
    pthread_cond_signal(&signal1);
    pre_task_flag[0] = 1;
    pthread_mutex_unlock(&mtx1);

    START_TIMER(time_start);
    slice_ptr = slice_arrays[1];
    omp_set_num_threads(1);
	// calc slice3/4/5
    #pragma omp parallel for
    for(int i = 0; i < data->msm_size; i++)
    {
        register uint64_t t1,t2,t0;
        t1 = input_scalars[i].dword[0] >> RIGHT_SHIFT_1;
        t2 = input_scalars[i].dword[1] << LEFT_SHIFT_1;

        slice_ptr[i] = signed_slices(t1 | t2, &carry_array[i], 0);
	}
    END_TIMER(time_end, time_start, time, "get slices batch1 3/4/5 cpu");
    pthread_mutex_lock(&mtx1);
    pre_task_flag[1] = 1;
    pthread_mutex_unlock(&mtx1);

    START_TIMER(time_start);
    slice_ptr = slice_arrays[2];
	// calc slice6/7/8
    #pragma omp parallel for
    for(int i = 0; i < data->msm_size; i++)
    {
        register uint64_t t1,t2,t0;
        t1 = input_scalars[i].dword[1] >> RIGHT_SHIFT_2;
        t2 = input_scalars[i].dword[2] << LEFT_SHIFT_2;

        slice_ptr[i] = signed_slices(t1 | t2, &carry_array[i], 0);
	}
    END_TIMER(time_end, time_start, time, "get slices batch2 6/7/8 cpu");
    pthread_mutex_lock(&mtx1);
    pre_task_flag[2] = 1;
    pthread_mutex_unlock(&mtx1);
	omp_set_num_threads(3);
    START_TIMER(time_start);
    slice_ptr = slice_arrays[3];
	// calc slice9/10/11
    #pragma omp parallel for
    for(int i = 0; i < data->msm_size; i++)
    {
        register uint64_t t1,t2,t0;
        t1 = input_scalars[i].dword[2] >> RIGHT_SHIFT_3;

        slice_ptr[i] = signed_slices(t1, &carry_array[i], 0);
	}
    END_TIMER(time_end, time_start, time, "get slices batch3 9/10/11 cpu");
    pthread_mutex_lock(&mtx1);
    pre_task_flag[3] = 1;
    pthread_mutex_unlock(&mtx1);

    START_TIMER(time_start);
    slice_ptr = slice_arrays[4];
	// calc slice12/13/14
    #pragma omp parallel for
    for(int i = 0; i < data->msm_size; i++)
    {
        register uint64_t t1,t2,t0;
        t1 = input_scalars[i].dword[3];

        slice_ptr[i] = signed_slices(t1, &carry_array[i], 0);
	}
    END_TIMER(time_end, time_start, time, "get slices batch4 12/13/14 cpu");
    pthread_mutex_lock(&mtx1);
    pre_task_flag[4] = 1;
    pthread_mutex_unlock(&mtx1);

    START_TIMER(time_start);
    slice_ptr = slice_arrays[5];
    // calc slice batch5_15/0/1      (msm0 + msm1)
    #pragma omp parallel for
    for(int i = 0; i < data->msm_size; i++)
    {
        register uint64_t t1,t2,t0;
        t1 = input_scalars[i].dword[3] >> RIGHT_SHIFT_1;
        t2 = input_scalars[i+data->msm_size].dword[0] << LEFT_SHIFT_1;

        slice_ptr[i] = signed_slices(t1 | t2, &carry_array[i], 1);
    }
    END_TIMER(time_end, time_start, time, "get slices batch5 15/0/1 cpu");
    pthread_mutex_lock(&mtx1);
    pre_task_flag[5] = 1;
    pthread_mutex_unlock(&mtx1);

    START_TIMER(time_start);
    slice_ptr = slice_arrays[6];
    // calc slice batch6_2/3/4       (msm1)
    #pragma omp parallel for
    for(int i = 0; i < data->msm_size; i++)
    {
        register uint64_t t1,t2,t0;
        t1 = input_scalars[i+data->msm_size].dword[0] >> RIGHT_SHIFT_2;
        t2 = input_scalars[i+data->msm_size].dword[1] << LEFT_SHIFT_2;

        slice_ptr[i] = signed_slices(t1 | t2, &carry_array[i], 0);
    }
    END_TIMER(time_end, time_start, time, "get slices batch6 2/3/4 cpu");
    pthread_mutex_lock(&mtx1);
    pre_task_flag[6] = 1;
    pthread_mutex_unlock(&mtx1);

    START_TIMER(time_start);
    slice_ptr = slice_arrays[7];
    // calc slice batch7_5/6/7       (msm1)
    #pragma omp parallel for
    for(int i = 0; i < data->msm_size; i++)
    {
        register uint64_t t1,t2,t0;
        t1 = input_scalars[i+data->msm_size].dword[1] >> RIGHT_SHIFT_3;

        slice_ptr[i] = signed_slices(t1, &carry_array[i], 0);
    }
    END_TIMER(time_end, time_start, time, "get slices batch7 5/6/7 cpu");
    pthread_mutex_lock(&mtx1);
    pre_task_flag[7] = 1;
    pthread_mutex_unlock(&mtx1);

    START_TIMER(time_start);
    slice_ptr = slice_arrays[8];
    // calc slice batch8_8/9/10       (msm1)
    #pragma omp parallel for
    for(int i = 0; i < data->msm_size; i++)
    {
        register uint64_t t1,t2,t0;
        t1 = input_scalars[i+data->msm_size].dword[2];

        slice_ptr[i] = signed_slices(t1, &carry_array[i], 0);
    }
    END_TIMER(time_end, time_start, time, "get slices batch8 8/9/10 cpu");
    pthread_mutex_lock(&mtx1);
    pre_task_flag[8] = 1;
    pthread_mutex_unlock(&mtx1);

    START_TIMER(time_start);
    slice_ptr = slice_arrays[9];
    // calc slice batch9_11/12/13       (msm1)
    #pragma omp parallel for
    for(int i = 0; i < data->msm_size; i++)
    {
        register uint64_t t1,t2,t0;
        t1 = input_scalars[i+data->msm_size].dword[2] >> RIGHT_SHIFT_1;
        t2 = input_scalars[i+data->msm_size].dword[3] << LEFT_SHIFT_1;

        slice_ptr[i] = signed_slices(t1 | t2, &carry_array[i], 0);
    }
    END_TIMER(time_end, time_start, time, "get slices batch9 11/12/13 cpu");
    pthread_mutex_lock(&mtx1);
    pre_task_flag[9] = 1;
    pthread_mutex_unlock(&mtx1);

    START_TIMER(time_start);
    slice_ptr = slice_arrays[10];
    // calc slice batch10_14/15/0       (msm1+msm2)
    #pragma omp parallel for
    for(int i = 0; i < data->msm_size; i++)
    {
        register uint64_t t1,t2,t0;
        t1 = input_scalars[i+data->msm_size].dword[3] >> RIGHT_SHIFT_2;
        t2 = input_scalars[i+data->msm_size*2].dword[0] << LEFT_SHIFT_2;

        slice_ptr[i] = signed_slices(t1 | t2, &carry_array[i], 2);
    }
    END_TIMER(time_end, time_start, time, "get slices batch10 14/15/0 cpu");
    pthread_mutex_lock(&mtx1);
    pre_task_flag[10] = 1;
    pthread_mutex_unlock(&mtx1);

    START_TIMER(time_start);
    slice_ptr = slice_arrays[11];
    // calc slice batch11_1/2/3       (msm2)
    #pragma omp parallel for
    for(int i = 0; i < data->msm_size; i++)
    {
        register uint64_t t1,t2;
        t1 = input_scalars[i+data->msm_size*2].dword[0] >> RIGHT_SHIFT_3;

        slice_ptr[i] = signed_slices(t1, &carry_array[i], 0);
    }
    END_TIMER(time_end, time_start, time, "get slices batch11 1/2/3 cpu");
    pthread_mutex_lock(&mtx1);
    pre_task_flag[11] = 1;
    pthread_mutex_unlock(&mtx1);

    START_TIMER(time_start);
    slice_ptr = slice_arrays[12];
    // calc slice batch12_4/5/6       (msm2)
    #pragma omp parallel for
    for(int i = 0; i < data->msm_size; i++)
    {
        register uint64_t t1,t2;
        t1 = input_scalars[i+data->msm_size*2].dword[1];

        slice_ptr[i] = signed_slices(t1, &carry_array[i], 0);
    }
    END_TIMER(time_end, time_start, time, "get slices batch12 4/5/6 cpu");
    pthread_mutex_lock(&mtx1);
    pre_task_flag[12] = 1;
    pthread_mutex_unlock(&mtx1);

    START_TIMER(time_start);
    slice_ptr = slice_arrays[13];
    // calc slice batch13_7/8/9       (msm2)
    #pragma omp parallel for
    for(int i = 0; i < data->msm_size; i++)
    {
        register uint64_t t1,t2;
        t1 = input_scalars[i+data->msm_size*2].dword[1] >> RIGHT_SHIFT_1;
        t2 = input_scalars[i+data->msm_size*2].dword[2] << LEFT_SHIFT_1;

        slice_ptr[i] = signed_slices(t1 | t2, &carry_array[i], 0);
    }
    END_TIMER(time_end, time_start, time, "get slices batch13 7/8/9 cpu");
    pthread_mutex_lock(&mtx1);
    pre_task_flag[13] = 1;
    pthread_mutex_unlock(&mtx1);

    START_TIMER(time_start);
    slice_ptr = slice_arrays[14];
    // calc slice batch14_10/11/12       (msm2)
    #pragma omp parallel for
    for(int i = 0; i < data->msm_size; i++)
    {
        register uint64_t t1,t2;
        t1 = input_scalars[i+data->msm_size*2].dword[2] >> RIGHT_SHIFT_2;
        t2 = input_scalars[i+data->msm_size*2].dword[3] << LEFT_SHIFT_2;

        slice_ptr[i] = signed_slices(t1 | t2, &carry_array[i], 0);
    }
    END_TIMER(time_end, time_start, time, "get slices batch14 10/11/12 cpu");
    pthread_mutex_lock(&mtx1);
    pre_task_flag[14] = 1;
    pthread_mutex_unlock(&mtx1);

    START_TIMER(time_start);
    slice_ptr = slice_arrays[15];
    // calc slice batch15_13/14/15       (msm2)
    #pragma omp parallel for
    for(int i = 0; i < data->msm_size; i++)
    {
        register uint64_t t1,t2;
        t1 = input_scalars[i+data->msm_size*2].dword[3] >> RIGHT_SHIFT_3;

        slice_ptr[i] = signed_slices(t1, &carry_array[i], 3);
    }
    END_TIMER(time_end, time_start, time, "get slices batch15 13/14/15 cpu");
    pthread_mutex_lock(&mtx1);
    pre_task_flag[15] = 1;
    pthread_mutex_unlock(&mtx1);

    START_TIMER(time_start);
    slice_ptr = slice_arrays[16];
    // calc slice batch16_0/1/2       (msm3)
    #pragma omp parallel for
    for(int i = 0; i < data->msm_size; i++)
    {
        register uint64_t t1,t2;
        t1 = input_scalars[i+data->msm_size*3].dword[0];

        slice_ptr[i] = signed_slices(t1, &carry_array[i], 0);
    }
    END_TIMER(time_end, time_start, time, "get slices batch16 0/1/2 cpu");
    pthread_mutex_lock(&mtx1);
    pre_task_flag[16] = 1;
    pthread_mutex_unlock(&mtx1);

    START_TIMER(time_start);
    slice_ptr = slice_arrays[17];
    // calc slice batch17_3/4/5       (msm3)
    #pragma omp parallel for
    for(int i = 0; i < data->msm_size; i++)
    {
        register uint64_t t1,t2;
        t1 = input_scalars[i+data->msm_size*3].dword[0] >> RIGHT_SHIFT_1;
        t2 = input_scalars[i+data->msm_size*3].dword[1] << LEFT_SHIFT_1;

        slice_ptr[i] = signed_slices(t1 | t2, &carry_array[i], 0);
    }
    END_TIMER(time_end, time_start, time, "get slices batch17 3/4/5 cpu");
    pthread_mutex_lock(&mtx1);
    pre_task_flag[17] = 1;
    pthread_mutex_unlock(&mtx1);

    START_TIMER(time_start);
    slice_ptr = slice_arrays[18];
    // calc slice batch18_6/7/8       (msm3)
    #pragma omp parallel for
    for(int i = 0; i < data->msm_size; i++)
    {
        register uint64_t t1,t2;
        t1 = input_scalars[i+data->msm_size*3].dword[1] >> RIGHT_SHIFT_2;
        t2 = input_scalars[i+data->msm_size*3].dword[2] << LEFT_SHIFT_2;

        slice_ptr[i] = signed_slices(t1 | t2, &carry_array[i], 0);
    }
    END_TIMER(time_end, time_start, time, "get slices batch18 6/7/8 cpu");
    pthread_mutex_lock(&mtx1);
    pre_task_flag[18] = 1;
    pthread_mutex_unlock(&mtx1);

    START_TIMER(time_start);
    slice_ptr = slice_arrays[19];
    // calc slice batch19_9/10/11       (msm3)
    #pragma omp parallel for
    for(int i = 0; i < data->msm_size; i++)
    {
        register uint64_t t1,t2;
        t1 = input_scalars[i+data->msm_size*3].dword[2] >> RIGHT_SHIFT_3;

        slice_ptr[i] = signed_slices(t1, &carry_array[i], 0);
    }
    END_TIMER(time_end, time_start, time, "get slices batch19 9/10/11 cpu");
    pthread_mutex_lock(&mtx1);
    pre_task_flag[19] = 1;
    pthread_mutex_unlock(&mtx1);

    START_TIMER(time_start);
    slice_ptr = slice_arrays[20];
    // calc slice batch20_12/13/14       (msm3)
    #pragma omp parallel for
    for(int i = 0; i < data->msm_size; i++)
    {
        register uint64_t t1,t2;
        t1 = input_scalars[i+data->msm_size*3].dword[3];

        slice_ptr[i] = signed_slices(t1, &carry_array[i], 0);
    }
    END_TIMER(time_end, time_start, time, "get slices batch20 12/13/14 cpu");
    pthread_mutex_lock(&mtx1);
    pre_task_flag[20] = 1;
    pthread_mutex_unlock(&mtx1);

    START_TIMER(time_start);
    slice_ptr = slice_arrays[21];
    // calc slice batch21_15/...       (msm3)
    #pragma omp parallel for
    for(int i = 0; i < data->msm_size; i++)
    {
        register uint64_t t1,t2;
        t1 = input_scalars[i+data->msm_size*3].dword[3] >> RIGHT_SHIFT_1;
        t2 = input_scalars[i+data->msm_size*3].dword[3] << LEFT_SHIFT_1; //just for avoiding 0

        slice_ptr[i] = signed_slices(t1 | t2, &carry_array[i], 3);
    }
    END_TIMER(time_end, time_start, time, "get slices batch21 15 cpu");

    pthread_mutex_lock(&mtx1);
    pre_task_flag[21] = 1;
    pthread_mutex_unlock(&mtx1);

    return nullptr;
}

struct PostThreadData
{
    char *output_results_hw_data;
    unsigned msm_batches_num;        // msm batches
    unsigned group_batches_size;     // group batches: 3 groups
    unsigned group_batches_num;      // the number of group batches for all msm bathes.
    unsigned groups_num_each_batch; 
    unsigned groups_num_all;
    unsigned bucket_num;
    unsigned *slice_bit;
    bool one_msm_done;
    vector<char *> output_batch_results_channel_vec;
    long long hw_output_batch_results_channel_bytes;
    vector<projective_sw_64bit> result_data;
};

projective_sw_64bit post_worker_thread_sub_task(projective_exed_64bit *groups, unsigned groups_num_each_batch, unsigned *slice_bit)
{
    unsigned long time = 0;
    struct timeval time_start, time_end;

    START_TIMER(time_start);
    projective_exed_64bit p_result = {};
    calc_groups_sum_64bit(&p_result, groups, groups_num_each_batch, slice_bit);
    END_TIMER(time_end, time_start, time, "Calc group sum cpu");

    START_TIMER(time_start);
    affine_exed_64bit result_affine_64bit = projective_exed_64bit::to_affine(p_result);

    affine_exed_64bit result_affine_neg_one_a_64bit = affine_exed_64bit::from_neg_one_a(result_affine_64bit); 

    affine_sw_64bit result_affine_sw_64bit = convert_exed_affine_to_sw_affine_64bit(result_affine_neg_one_a_64bit); 
    END_TIMER(time_end, time_start, time, "MSM Projective EXED -> Affine SW point cpu");

    projective_sw_64bit result_projective_sw_64bit;
    result_projective_sw_64bit.x = result_affine_sw_64bit.x;
    result_projective_sw_64bit.y = result_affine_sw_64bit.y;
    result_projective_sw_64bit.z = fp_64bit::get_field_one(); 

    #if DEBUG_INFO
    affine_exed_64bit::print(result_affine_64bit);
    affine_exed_64bit::print(result_affine_neg_one_a_64bit);
    affine_sw_64bit::print(result_affine_sw_64bit);
    projective_sw_64bit::print(result_projective_sw_64bit);
    #endif

    return result_projective_sw_64bit;
}

void *post_worker_thread(void *arg)
{
    printf("      [HOST][%s] launch pthread\n", __FUNCTION__);

    PostThreadData *data = static_cast<PostThreadData *>(arg);
    data->one_msm_done = false;
    data->result_data.resize(data->msm_batches_num);
    projective_exed_64bit *groups = new projective_exed_64bit[data->groups_num_all];
    unsigned long time = 0;
    struct timeval time_start, time_end;

    unsigned count = 0;

    // for (int i = 0; i < data->group_batches_num; i++)
    while (true)
    {
        // printf("      [HOST][%s] loop start[%d]\n", __FUNCTION__, count);
        /* code */
        // pthread_mutex_lock(&mtx2);
        // pthread_cond_wait(&signal2, &mtx2);
        // pthread_mutex_unlock(&mtx2);

        pthread_mutex_lock(&mtx2);
        if (!post_task_flag[count])
        {
            pthread_mutex_unlock(&mtx2);
            continue;
        }
        pthread_mutex_unlock(&mtx2);


        usleep(10000);

        START_TIMER(time_start);
        for (int k = 0; k < data->group_batches_size; k++)
        {
            memcpy( data->output_results_hw_data +
                    data->hw_output_batch_results_channel_bytes * data->group_batches_size * count + data->hw_output_batch_results_channel_bytes * k,
                    data->output_batch_results_channel_vec[k],
                    data->hw_output_batch_results_channel_bytes);
        }
        END_TIMER(time_end, time_start, time, "post memcpy cpu");

        unsigned group_start = count * data->group_batches_size;
        unsigned group_end = count * data->group_batches_size + data->group_batches_size - 1;

        if (group_end >= data->groups_num_all - 1)
        {
            group_end = data->groups_num_all - 1;
        }

        START_TIMER(time_start);
        msm_batch_calc_group_post_process_64bit(groups,
                                                data->output_results_hw_data,
                                                group_start,
                                                group_end,
                                                data->bucket_num,
                                                data->slice_bit);
        END_TIMER(time_end, time_start, time, "msm_batch_calc_group_post_process_64bit cpu");

        pthread_mutex_lock(&mtx2);
        if (data->one_msm_done)
        {
            pthread_mutex_unlock(&mtx2);
            #if PROFILING_INFO
            printf("    \n      [HOST][%s] Get one_msm_done[%d]\n", __FUNCTION__, count);
            #endif
            data->one_msm_done = false;
            break;
        }
        pthread_mutex_unlock(&mtx2);
        count++;

        // printf("      [HOST][%s] loop end[%d]\n", __FUNCTION__, count);
    }

    #if DEBUG_INFO
    for(int i = 0; i < data->groups_num_all; i++)
    {
        printf("    @@ group[%d]: \n", i);
        projective_exed_64bit::print(groups[i]);
    }
    #endif

    for (int i = 0; i < data->msm_batches_num; i++)
    {
        // printf("      [HOST][%s] batch: %d\n", __FUNCTION__, i); 
        data->result_data[i] = post_worker_thread_sub_task( groups + data->groups_num_each_batch * i,
                                                            data->groups_num_each_batch,
                                                            data->slice_bit);
    }

    pthread_mutex_lock(&mtx2);
    pthread_cond_signal(&signal3);
    pthread_mutex_unlock(&mtx2);

    delete groups;
}

class Driver
{
public:
    Driver(ssize_t npoints);
    ~Driver();

    int init(const void *rust_points);
    int load_xclbin(const std::string &binaryFile);
    int run_msm(void *out, uint64_t msm_batches_num, void *ptr_scalars);

    void deinit();

 private:
    cl_int err;
    cl::CommandQueue q, b_q;
    cl::Context context;
    cl::Kernel krnl_msm;

    std::vector<cl::Buffer> buffer_input_points;
    std::vector<cl::Buffer> buffer_input_scalars;
    std::vector<cl::Buffer> buffer_output_results;

    // Allocate points memory with alignment
    std::vector<char, aligned_allocator<char>> input_points_channel_data0;
    std::vector<char, aligned_allocator<char>> input_points_channel_data1;
    std::vector<char, aligned_allocator<char>> input_points_channel_data2;
    std::vector<char, aligned_allocator<char>> input_points_channel_data3;
    // Allocate scalars memory with alignment
    std::vector<uint16_t, aligned_allocator<uint16_t>> input_batch_slices_data0;
    std::vector<uint16_t, aligned_allocator<uint16_t>> input_batch_slices_data1; 
    // Allocate results memory
    std::vector<char, aligned_allocator<char>> output_batch_results_channel_data0;
    std::vector<char, aligned_allocator<char>> output_batch_results_channel_data1;
    std::vector<char, aligned_allocator<char>> output_batch_results_channel_data2;

    int bases_data_channels_num;
    long long input_sw_points_total_bytes;
    long long input_exed_points_total_bytes;
    long long hw_input_bases_channel_bytes;
    long long hw_input_batch_slices_bytes;
    long long hw_output_batch_results_channel_bytes;
    long long input_all_slices_bytes;
    long long output_results_hw_bytes;

    int bit_s;
    int groups_num_each_batch;
    int groups_num_all;
    unsigned bucket_num;
    std::vector<unsigned> slice_bit;
    int k;
    int msm_size;
    int group_batches_size;
    int group_batches_num;

    uint16_t *input_all_slices_data;

    vector<uint64_t> slice_array;
    char *output_results_hw_data;

};

Driver::Driver(ssize_t npoints)
{
    msm_size = npoints;
    k = std::log2(msm_size);
    bit_s = get_bit_s();
    groups_num_each_batch = get_group_number_64bit(bit_s);

    #if HW_0_BUCKET
    bucket_num = 1 << (bit_s);
    #else
    bucket_num = 1 << (bit_s - 1);
    #endif

    slice_bit.resize(groups_num_each_batch);
    get_slice_bit_64bit(slice_bit.data(), groups_num_each_batch, bit_s);

    bases_data_channels_num = HW_BASES_DATA_CHANNEL_NUM;
    group_batches_size = HW_BATCH_GROUPS_NUM;
    group_batches_num = (groups_num_each_batch + group_batches_size - 1) / group_batches_size;
    printf("      [HOST] k: %d, msm_size: %d, bit_s: %d, groups_num_each_batch: %d, bucket_num: %d\n", k, msm_size, bit_s, groups_num_each_batch, bucket_num);
    printf("      [HOST] group_batches_size: %d, group_batches_num: %d, bases_data_channels_num: %d\n", group_batches_size, group_batches_num, bases_data_channels_num);


    int loop_count = msm_size / bases_data_channels_num;
    input_points_channel_data0.resize(loop_count * EXED_AFFINE_BYTES);
    input_points_channel_data1.resize(loop_count * EXED_AFFINE_BYTES);
    input_points_channel_data2.resize(loop_count * EXED_AFFINE_BYTES);
    input_points_channel_data3.resize(loop_count * EXED_AFFINE_BYTES);

    // points
    input_sw_points_total_bytes = SW_AFFINE_BYTES * (long long)msm_size;
    input_exed_points_total_bytes = EXED_AFFINE_BYTES * (long long)msm_size;
    hw_input_bases_channel_bytes = input_exed_points_total_bytes / bases_data_channels_num;

    // scalars
    hw_input_batch_slices_bytes = msm_size * HW_BATCH_GROUPS_CHANNEL_NUM * sizeof(uint16_t);
    input_batch_slices_data0.resize(msm_size * HW_BATCH_GROUPS_CHANNEL_NUM);
    input_batch_slices_data1.resize(msm_size * HW_BATCH_GROUPS_CHANNEL_NUM);

    slice_array.resize(msm_size * GROUP_BATCHES_NUM_MAX);

    // result
    // (sum, temp), segment = 128,
    hw_output_batch_results_channel_bytes = EXED_PROJECTIVE_BYTES * 2 * HW_CALC_GROUP_SPLIT_SUB_GROUP_NUM_FOR_ZPRIZE_2023; 

    #if DEBUG_INFO
    printf("      [HOST] input_sw_points_total_bytes: %ld\n", input_sw_points_total_bytes);
    printf("      [HOST] input_exed_points_total_bytes: %ld\n", input_exed_points_total_bytes);
    printf("      [HOST] hw_input_bases_channel_bytes: %ld\n", hw_input_bases_channel_bytes);
    printf("      [HOST] hw_input_batch_slices_bytes: %ld\n", hw_input_batch_slices_bytes);
    printf("      [HOST] hw_output_batch_results_channel_bytes: %ld\n", hw_output_batch_results_channel_bytes);
    #endif

    output_batch_results_channel_data0.resize(hw_output_batch_results_channel_bytes);
    output_batch_results_channel_data1.resize(hw_output_batch_results_channel_bytes);
    output_batch_results_channel_data2.resize(hw_output_batch_results_channel_bytes);

    // (sum, temp), segment = 128, group_num = 16
    output_results_hw_bytes = EXED_PROJECTIVE_BYTES * 2 * HW_CALC_GROUP_SPLIT_SUB_GROUP_NUM_FOR_ZPRIZE_2023 * groups_num_each_batch; 

    output_results_hw_data = (char *)malloc(output_results_hw_bytes * 16 * 4);       // msm batch= 4
    if (output_results_hw_data == NULL)
    {
        printf("\n      [HOST] Error: Allocate output_results_hw_data failed!\n");
        return;
    }
}

Driver::~Driver()
{
    
}

int Driver::init(const void *rust_points)
{
    // Load from Rust
    #if LOAD_FROM_RUST
    affine_sw_64bit *sw_points = (affine_sw_64bit *)malloc(msm_size * sizeof(affine_sw_64bit));
    for(int i = 0; i < msm_size; i++)
    {
        memcpy((void *)sw_points + SW_AFFINE_BYTES * i, rust_points + (SW_AFFINE_BYTES + 8) * i, SW_AFFINE_BYTES);
    }
    #else // Load from C/C++
    affine_sw_64bit *sw_points = (affine_sw_64bit *)rust_points;
    #endif

    // SW -> EXED
    affine_exed_64bit *exed_points = (affine_exed_64bit *)malloc(msm_size * sizeof(affine_exed_64bit));

    #if DEBUG_INFO
    printf("      [HOST][%s] SW point \n", __FUNCTION__);
    affine_sw_64bit::print(sw_points[0]);
    affine_sw_64bit::print(sw_points[msm_size - 1]);
    affine_sw_64bit swp1 = affine_sw_64bit::to_montgomery(sw_points[0]);
    affine_sw_64bit swp2 = affine_sw_64bit::to_montgomery(sw_points[msm_size - 1]);
    printf("      [HOST][%s] sw to_montgomery\n", __FUNCTION__);
    affine_sw_64bit::print(swp1);
    affine_sw_64bit::print(swp2);
    #endif

    printf("      [HOST][%s] Convert SW -> EXED points...\n", __FUNCTION__);
    #pragma omp parallel for
    for(int i = 0; i < msm_size; i++)
    {
        exed_points[i] = convert_sw_affine_to_exed_affine_64bit(sw_points[i]);
    }

    printf("      [HOST][%s] Precompute EXED points...\n", __FUNCTION__);
    // Precompute
    CURVE_EXED::precompute_exed_bases_points_for_64bit(exed_points, exed_points, msm_size);

    #if DEBUG_INFO
    printf("      [HOST][%s] EXED point \n", __FUNCTION__);
    affine_exed_64bit::print(exed_points[0]);
    affine_exed_64bit::print(exed_points[msm_size - 1]);
    #endif

    // Split channels
    int loop_count = msm_size / bases_data_channels_num;
    // printf("      [HOST][%s] npoints: %d, bases_data_channels_num: %d, loop_count: %d\n", __FUNCTION__, msm_size, bases_data_channels_num, loop_count);

    char *exed_points_ptr = (char *)exed_points;

    // int num_threads = 8;
    // omp_set_num_threads(num_threads);
    printf("      [HOST][%s] Split EXED points for FPGA...\n", __FUNCTION__);
    #pragma omp parallel for
    for (long long i = 0; i < loop_count; i++)
    {
        long long offset = i * EXED_AFFINE_BYTES * bases_data_channels_num;
        long long offset1 = i * EXED_AFFINE_BYTES;

        for (long long j = 0; j < EXED_AFFINE_BYTES; j++) {
            input_points_channel_data0[offset1 + j] = exed_points_ptr[offset + j];
        }

        for (long long j = 0; j < EXED_AFFINE_BYTES; j++) {
            input_points_channel_data1[offset1 + j] = exed_points_ptr[offset + j + EXED_AFFINE_BYTES];
        }

        for (long long j = 0; j < EXED_AFFINE_BYTES; j++) {
            input_points_channel_data2[offset1 + j] = exed_points_ptr[offset + j + 2 * EXED_AFFINE_BYTES];
        }

        for (long long j = 0; j < EXED_AFFINE_BYTES; j++) {
            input_points_channel_data3[offset1 + j] = exed_points_ptr[offset + j + 3 * EXED_AFFINE_BYTES];
        }
    }

    #if LOAD_FROM_RUST
    if (sw_points)
    {
        free(sw_points);
    }
    #endif

    if (exed_points)
    {
        free(exed_points);
    }

    return 0;
}

int Driver::load_xclbin(const std::string &binaryFile)
{
    unsigned long time = 0;
    struct timeval time_start, time_end;

    // Create Program and Kernel
    auto devices = xcl::get_xil_devices();

    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++)
    {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, 0, &err));
        OCL_CHECK(err, b_q = cl::CommandQueue(context, device, 0, &err));

        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS)
        {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        }
        else
        {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_msm = cl::Kernel(program, "krnl_msm_rtl", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    
        cl_device_id device_id = device();
        cl_ulong max_mem_alloc_size;
        err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_mem_alloc_size, NULL);
        if (err != CL_SUCCESS)
        {
            std::cerr << "Error getting max memory allocation size" << std::endl;
        } else
        {
            std::cout << "Max memory allocation size: " << max_mem_alloc_size << " bytes" << std::endl;
        }
    }
    if (!valid_device)
    {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    START_TIMER(time_start);
    // Allocate Buffer in Global Memory
    // input source
    OCL_CHECK(err, buffer_input_points.emplace_back(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hw_input_bases_channel_bytes,
                                        input_points_channel_data0.data(), &err));
    OCL_CHECK(err, buffer_input_points.emplace_back(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hw_input_bases_channel_bytes,
                                        input_points_channel_data1.data(), &err));
    OCL_CHECK(err, buffer_input_points.emplace_back(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hw_input_bases_channel_bytes,
                                        input_points_channel_data2.data(), &err));
    OCL_CHECK(err, buffer_input_points.emplace_back(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hw_input_bases_channel_bytes,
                                        input_points_channel_data3.data(), &err));

    OCL_CHECK(err, buffer_input_scalars.emplace_back(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hw_input_batch_slices_bytes,
                                        input_batch_slices_data0.data(), &err));
    OCL_CHECK(err, buffer_input_scalars.emplace_back(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, hw_input_batch_slices_bytes,
                                        input_batch_slices_data1.data(), &err));

    OCL_CHECK(err, buffer_output_results.emplace_back(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, hw_output_batch_results_channel_bytes,
                                       output_batch_results_channel_data0.data(), &err));
    OCL_CHECK(err, buffer_output_results.emplace_back(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, hw_output_batch_results_channel_bytes,
                                       output_batch_results_channel_data1.data(), &err));
    OCL_CHECK(err, buffer_output_results.emplace_back(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, hw_output_batch_results_channel_bytes,
                                       output_batch_results_channel_data2.data(), &err));

    // Set the Kernel Arguments
    OCL_CHECK(err, err = krnl_msm.setArg(0, buffer_input_points[0]));    // bases channel 0
    OCL_CHECK(err, err = krnl_msm.setArg(1, buffer_input_points[1]));    // bases channel 1
    OCL_CHECK(err, err = krnl_msm.setArg(2, buffer_input_scalars[0]));    // scalar
    OCL_CHECK(err, err = krnl_msm.setArg(3, buffer_input_points[2]));    // bases channel 2
    OCL_CHECK(err, err = krnl_msm.setArg(4, buffer_input_points[3]));    // bases channel 3
    OCL_CHECK(err, err = krnl_msm.setArg(5, buffer_output_results[0]));    // result channel 0
    OCL_CHECK(err, err = krnl_msm.setArg(6, buffer_output_results[1]));    // result channel 1
    OCL_CHECK(err, err = krnl_msm.setArg(7, buffer_output_results[2]));    // result channel 2
    OCL_CHECK(err, err = krnl_msm.setArg(8, msm_size));
    OCL_CHECK(err, err = krnl_msm.setArg(9, buffer_input_scalars[1]));    // scalar
    int scalar_switch_flag = 0;
    OCL_CHECK(err, err = krnl_msm.setArg(10, scalar_switch_flag));        //which scalar buffer  

    END_TIMER(time_end, time_start, time, "Allocate Buffer in Global Memory and Set kernel");

    START_TIMER(time_start);
    // Load points into the FPGA
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_input_points[0], buffer_input_points[1], buffer_input_points[2], buffer_input_points[3]}, 0 /* 0 means from host*/));
    END_TIMER(time_end, time_start, time, "Load points into the FPGA");

    return 0;
}

int Driver::run_msm(void *out, uint64_t msm_batches_num, void *ptr_scalars)
{
    unsigned long time = 0;
    struct timeval time_start, time_end;
    struct timeval time_start1, time_end1;
    struct timeval time_start2, time_end2;

    // Reset task flag
    for (int i = 0; i < GROUP_BATCHES_NUM_MAX; i++)
    {
        pre_task_flag[i] = 0;
        post_task_flag[i] = 0;
    }

    static int run_msm_count = 0;
    run_msm_count++;

    PostThreadData post_thread_data;
    post_thread_data.output_results_hw_data = output_results_hw_data;
    post_thread_data.group_batches_size = group_batches_size;
    post_thread_data.groups_num_each_batch = groups_num_each_batch;
    groups_num_all = groups_num_each_batch * msm_batches_num;
    group_batches_num = (groups_num_all + group_batches_size - 1) / group_batches_size;
    printf("      [HOST][%s] all msm_batches_num: %d, groups_num_all: %d, group_batches_size: %d, group_batches_num: %d\n",
            __FUNCTION__, msm_batches_num, groups_num_all, group_batches_size, group_batches_num);

    post_thread_data.msm_batches_num = msm_batches_num;
    post_thread_data.group_batches_num = group_batches_num;
    post_thread_data.groups_num_all = groups_num_all;
    post_thread_data.bucket_num = bucket_num;
    post_thread_data.slice_bit = slice_bit.data();
    post_thread_data.one_msm_done = false;
    vector<char *>output_batch_results_channel_vec_tmp(3);
    output_batch_results_channel_vec_tmp[0] = output_batch_results_channel_data0.data();
    output_batch_results_channel_vec_tmp[1] = output_batch_results_channel_data1.data();
    output_batch_results_channel_vec_tmp[2] = output_batch_results_channel_data2.data();
    post_thread_data.output_batch_results_channel_vec = output_batch_results_channel_vec_tmp;
    post_thread_data.hw_output_batch_results_channel_bytes = hw_output_batch_results_channel_bytes;

    pthread_t post_worker_handler;
    if (pthread_create(&post_worker_handler, nullptr, post_worker_thread, &post_thread_data) != 0)
    {
        std::cerr << "      [HOST] Error creating post_worker_thread." << std::endl;
        return -1;
    }

    // pre_worker thread
    PreThreadData pre_thread_data;
    pre_thread_data.input_scalars = (char *)ptr_scalars;
    pre_thread_data.all_slices = slice_array.data();
    pre_thread_data.msm_size = msm_size;
    pre_thread_data.group_batches_num = group_batches_num;
    pre_thread_data.groups_num_each_batch = groups_num_each_batch;
    pre_thread_data.bit_s = bit_s;
    pre_thread_data.slice_bit = slice_bit.data();

    START_TIMER(time_start);
    pthread_t pre_worker_handler;
    if (pthread_create(&pre_worker_handler, nullptr, pre_worker_thread, &pre_thread_data) != 0)
    {
        std::cerr << "      [HOST] Error creating pre_worker_thread." << std::endl;
        return -1;
    }

    // Wait for signal1
    pthread_mutex_lock(&mtx1);
    pthread_cond_wait(&signal1, &mtx1);
    pthread_mutex_unlock(&mtx1);
    END_TIMER(time_end, time_start, time, "Get first slices cpu");

    printf("      [HOST][%s] Main thread received notification --->\n", __FUNCTION__);

    for (int i = 0; i < group_batches_num; i++)
    {
        #if PROFILING_INFO
        printf("\n\n      [HOST][%s]  ============================  Kernel loop %d ============================\n", __FUNCTION__, i);
        #endif

        START_TIMER(time_start1);
        OCL_CHECK(err, err = krnl_msm.setArg(10, i%2 == 0 ? 0 : 1));

        if(i == 0)
        {
            START_TIMER(time_start);
            memcpy( input_batch_slices_data0.data(),
                    (uint16_t *)slice_array.data() + i * HW_BATCH_GROUPS_CHANNEL_NUM * msm_size,
                    hw_input_batch_slices_bytes);
            END_TIMER(time_end, time_start, time, "Pre-process memcpy the first memory");

            START_TIMER(time_start);
            // Copy input data to device global memory
            OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_input_scalars[0]}, 0 /* 0 means from host*/));
            OCL_CHECK(err, err = q.finish());
            END_TIMER(time_end, time_start, time, "Pre-process copy the first input data to device global memory");
 
        }

        #if DEBUG_INFO
        for(int i = 0; i < 4; i++)
        {
            printf("    [HOST][%s] input_batch_slices_data[%d]: 0x%x\n", __FUNCTION__, i, input_batch_slices_data[i]);
        }
        #endif

        #if PROFILING_INFO
        printf("      [HOST][%s] Call kernel\n", __FUNCTION__);
        #endif

        // Copy input data to device global memory
        //OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_input_scalars[0]}, 0 /* 0 means from host*/));
        // Launch the Kernel
        OCL_CHECK(err, err = q.enqueueTask(krnl_msm));
        // Copy Result from Device Global Memory to Host Local Memory
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({   buffer_output_results[0],
                                                            buffer_output_results[1],
                                                            buffer_output_results[2]},
                                                            CL_MIGRATE_MEM_OBJECT_HOST));
        usleep(15000);

        if (i < group_batches_num -1)
        {
            while (true)
            {
                pthread_mutex_lock(&mtx1);
                if (!pre_task_flag[i + 1])
                {
                    pthread_mutex_unlock(&mtx1);
                    continue;
                }
                pthread_mutex_unlock(&mtx1);

                START_TIMER(time_start);
                if(i % 2 == 0)
                {    
                    memcpy( input_batch_slices_data1.data(),
                            (uint16_t *)slice_array.data() + (i+1) * HW_BATCH_GROUPS_CHANNEL_NUM * msm_size,
                            hw_input_batch_slices_bytes);
                    OCL_CHECK(err, err = b_q.enqueueMigrateMemObjects({buffer_input_scalars[1]}, 0 /* 0 means from host*/));
                };

                if(i % 2 == 1)
                {    
                    memcpy( input_batch_slices_data0.data(),
                            (uint16_t *)slice_array.data() + (i+1) * HW_BATCH_GROUPS_CHANNEL_NUM * msm_size,
                            hw_input_batch_slices_bytes);
                    OCL_CHECK(err, err = b_q.enqueueMigrateMemObjects({buffer_input_scalars[0]}, 0 /* 0 means from host*/));
                };
                END_TIMER(time_end, time_start, time, "Memcpy and copy the next input data to device global memory");
                break;
            }
        }

        OCL_CHECK(err, err = b_q.finish());
        OCL_CHECK(err, err = q.finish());

        //sleep(2);

        END_TIMER(time_end1, time_start1, time, "Running kernel one batch");

        #if PROFILING_INFO
        printf("      [HOST][%s] Call kernel done\n\n", __FUNCTION__);
        #endif

        pthread_mutex_lock(&mtx2);
        // pthread_cond_signal(&signal2);
        post_task_flag[i] = 1;
        pthread_mutex_unlock(&mtx2);

        if (i == group_batches_num -1)
        {
            pthread_mutex_lock(&mtx2);
            #if PROFILING_INFO
            printf("      [HOST][%s] Send post_thread_data.one_msm_done\n", __FUNCTION__);
            #endif
            post_thread_data.one_msm_done = true;
            pthread_mutex_unlock(&mtx2);
        }
    }

    // deinit
    pthread_mutex_lock(&mtx2);
    pthread_cond_wait(&signal3, &mtx2);
    pthread_mutex_unlock(&mtx2);
    // post_thread_data.result_data.resize(msm_batches_num);
    memcpy(out, post_thread_data.result_data.data(), SW_PROJECTIVE_BYTES * msm_batches_num);

    // printf("      [HOST][%s] pthread_join\n", __FUNCTION__);
    pthread_cancel(pre_worker_handler);
    pthread_join(pre_worker_handler, nullptr);
    pthread_cancel(post_worker_handler);
    pthread_join(post_worker_handler, nullptr);

    printf("      [HOST][%s] Run msm completed\n\n\n", __FUNCTION__);
    return 0;
}

void Driver::deinit()
{
    pthread_mutex_destroy(&mtx1);
    pthread_mutex_destroy(&mtx2);
    pthread_cond_destroy(&signal1);
    pthread_cond_destroy(&signal2);
    pthread_cond_destroy(&signal3);
    pthread_cond_destroy(&signal4);

    if (output_results_hw_data)
    {
        free(output_results_hw_data);
    }
}

extern "C" Driver *msm_init_for_ss(const char *xclbin, ssize_t xclbin_len, const void *rust_points, ssize_t npoints)
{
    std::cout << "\n\nInstantiating msm driver for " << npoints << " points"
                << std::endl;
    auto *driver = new Driver(npoints);

    printf("      [HOST][%s] Driver init\n", __FUNCTION__);
    int ret = 0;
    ret = driver->init(rust_points);
    if (ret)
    {
        std::cout << "Error: driver init failed!" << std::endl;
        return NULL;
    }

    printf("      [HOST][%s] Driver load_xclbin\n", __FUNCTION__);

    std::string binaryFile(xclbin, xclbin_len);

    std::cout << "Loading xclbin=" << xclbin << std::endl;
    std::cout << "Loading XCLBIN=" << binaryFile << std::endl;

    ret = driver->load_xclbin(binaryFile);

    if (ret)
    {
        std::cout << "Error: driver load xclbin failed!" << std::endl;
        driver->deinit();
        return NULL;
    }

    // printf("      [HOST][%s] Completed\n", __FUNCTION__);
    return driver;
}

extern "C" void msm_mult_for_ss(Driver *driver,void *out, uint64_t batches_num, void *ptr_scalars)
{
    int ret = driver->run_msm(out, batches_num, ptr_scalars);

    if (ret)
    {
        std::cout << "Error: driver run msm failed!" << std::endl;
        driver->deinit();
        return;
    }
}

#define TEST_GOLDEN_BASES_PATH "../data/size8192-c7-1/bases.bin"
#define TEST_GOLDEN_SCALARS_PATH "../data/size8192-c7-1/scalars.bin"
int run_msm_test(int argc, char **argv)
{
    int ret = 0;
    // load test data
    auto k = 13;
    auto npoints = 1 << k;

    long long input_bases_bytes = SW_AFFINE_BYTES * (long long)npoints;
    std::vector<char>input_bases_data(input_bases_bytes);

    ret = load_data(input_bases_data.data(), TEST_GOLDEN_BASES_PATH, SW_AFFINE_BYTES, npoints, 0);
    if (ret)
    {
        printf("      [HOST] Error: load bases data failed\n");
        return -1;
    }

    auto input_scalars_bytes = FR_BYTES * npoints;
    std::vector<char>input_scalars_data(input_bases_bytes);

    ret = load_data(input_scalars_data.data(), TEST_GOLDEN_SCALARS_PATH, FR_BYTES, npoints, 0);
    if (ret)
    {
        printf("      [HOST] Error: load scalars data failed\n");
        return -1;
    }

    #if DEBUG_INFO
    for (int i = 0; i < 32; i++)
    {
        printf("    @@ input_bases_data[%d]: 0x%x\n", i, (unsigned char)input_bases_data[i]);
    }
    for (int i = 0; i < 32; i++)
    {
        printf("    @@ input_scalars_data[%d]: 0x%x\n", i, (unsigned char)input_scalars_data[i]);
    }
    #endif

    auto *driver = new Driver(npoints);

    ret = driver->init((void *)input_bases_data.data());
    if (ret)
    {
        std::cout << "Error: driver init failed!" << std::endl;
        return -1;
    }

    std::string binaryFile = argv[1];
    std::cout << "Loading XCLBIN=" << binaryFile << std::endl;

    ret = driver->load_xclbin(binaryFile);

    if (ret)
    {
        std::cout << "Error: driver load xclbin failed!" << std::endl;
        driver->deinit();
        return -1;
    }

    // std::vector<char>out(1);
    // ret = driver->run_msm((void *)out.data(), (void *)input_scalars_data.data());
    ret = driver->run_msm(NULL, 1, (void *)input_scalars_data.data());
    if (ret)
    {
        std::cout << "Error: driver run msm failed!" << std::endl;
        driver->deinit();
        return -1;
    }

    delete driver;

    return 0;
} 

int main(int argc, char **argv)
{

    run_msm_test(argc, argv);

    return 0;
}

