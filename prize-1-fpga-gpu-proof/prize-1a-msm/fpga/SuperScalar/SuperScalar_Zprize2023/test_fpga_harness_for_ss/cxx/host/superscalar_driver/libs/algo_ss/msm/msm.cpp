#include <stdio.h>
#include <cstring>
#include <vector>
#include <cmath>
#include <omp.h>

#include "common.h"
#include "math.h"
#include "msm.h"
#include "config.h"
#include "curve/curve.h"

using namespace std;
using namespace CURVE_BLS12_377;
using namespace common;

namespace msm_cmodel
{

    template <typename Fr>
    static unsigned get_group_number(unsigned bit_s)
    {
        return (Fr::BC + (bit_s - 1)) / bit_s;
    }

    unsigned get_bit_s()
    {
        return BIT_S;
    }

    template <typename Fr>
    void get_slices_info(unsigned groups_num, unsigned bit_s, vector<unsigned> &index_start, vector<unsigned> &index_end, vector<unsigned> &m, vector<unsigned> &n, vector<unsigned> &m_shift, vector<unsigned> &n_shift)
    {
        for (int i = 0; i < groups_num; i++)
        {
            unsigned index_start_tmp = i * bit_s;
            unsigned index_end_tmp = (i < (groups_num - 1)) ? ((i + 1) * bit_s - 1) : (Fr::BC - 1);
            unsigned m_tmp = (Fr::LB == 0) ? 0 : (index_start_tmp / (Fr::LB));
            unsigned n_tmp = (Fr::LB == 0) ? 0 : (index_end_tmp / (Fr::LB));
            unsigned m_shift_tmp = index_start_tmp - Fr::LB * m_tmp;
            unsigned n_shift_tmp = index_end_tmp - Fr::LB * n_tmp;
            index_start.push_back(index_start_tmp);
            index_end.push_back(index_end_tmp);
            m.push_back(m_tmp);
            n.push_back(n_tmp);
            m_shift.push_back(m_shift_tmp);
            n_shift.push_back(n_shift_tmp);
        }

    }

    template <typename Projective>
    static int msm_batch_calc_group_post_process(Projective *groups, Projective *segments_results, unsigned groups_start, unsigned groups_end, unsigned buckets_num, unsigned sub_group_num)
    {
        int tri_segments_num = sub_group_num;
        int rec_segments_num = sub_group_num - 1;
        int segment_len = (buckets_num + sub_group_num - 1) / sub_group_num;

        // printf("      [HOST][%s] case, segment_len: %d\n", __FUNCTION__, segment_len);
        int num_threads = 8;
        // printf("      [HOST]num_threads: %d\n", num_threads);
        omp_set_num_threads(num_threads);
        #pragma omp parallel for
        for (unsigned i = groups_start; i <= groups_end; i++)
        {
            vector<int> tri_segments_len;
            vector<int> tri_segments_rav_offest;

            int offest = 0;
            for (int j = 0; j < tri_segments_num; j++)
            {
                int tri_segment_len = segment_len;

                #if HW_0_BUCKET
                if (j == 0)
                {
                    tri_segment_len = tri_segment_len -1;
                }
                #endif

                // Consider the problem that bucket num is not divisible
                if (j == (tri_segments_num - 1))
                {
                    tri_segment_len = buckets_num - j * segment_len;
                }

                tri_segments_len.push_back(tri_segment_len);
                
                #if HW_0_BUCKET
                if (j == 0)
                {
                    tri_segment_len = segment_len;
                }
                #endif

                offest += tri_segment_len;
                tri_segments_rav_offest.push_back((offest - 1));
            }

            Projective *tri_segments_sum_group = segments_results + tri_segments_num * 2 * i;

            vector<Projective> tri_segments_sum;
            vector<Projective> tri_segments_running_sum;
            for (int j = 0; j < tri_segments_num; j++)
            {
                tri_segments_sum.push_back(tri_segments_sum_group[j * 2]);
                tri_segments_running_sum.push_back(tri_segments_sum_group[j * 2 + 1]);
            }

            // Step 2 calc temp
            vector<Projective> rec_segments_side_sum;
            Projective sum2;
            Projective::init(sum2);
            for (int j = 0; j < rec_segments_num; j++)
            {
                sum2 = sum2 + tri_segments_running_sum[tri_segments_num - 1 - j];
                rec_segments_side_sum.push_back(sum2);
            }

            // Step 3
            vector<Projective> rec_segments_values;
            for (int j = 0; j < rec_segments_num; j++)
            {
                int count = log2(tri_segments_len[j]);

                Projective sum3 = rec_segments_side_sum[rec_segments_num - 1 - j];

                #if HW_0_BUCKET
                if (j == 0)
                {
                    Projective sum3_tmp;
                    Projective::init(sum3_tmp);
                    for (int k = 0; k < tri_segments_len[j]; k++)
                    {
                        sum3_tmp = sum3_tmp + sum3;
                    }
                    sum3 = sum3_tmp;
                }
                else
                {
                    for (int k = 0; k < count; k++)
                    {
                        sum3 = sum3 + sum3;
                    }
                }
                #else
                for (int k = 0; k < count; k++)
                {
                    sum3 = sum3 + sum3;
                }
                #endif

                rec_segments_values.push_back(sum3);
            }

            // Step 4
            Projective sum;
            Projective::init(sum);
            for (int j = 0; j < tri_segments_sum.size(); j++)
            {
                // For more hardware parallelism
                tri_segments_sum[j] = tri_segments_sum[j] + tri_segments_running_sum[j];

                sum = sum + tri_segments_sum[j];
            }
            for (int j = 0; j < rec_segments_values.size(); j++)
            {
                sum = sum + rec_segments_values[j];
            }

            groups[i] = sum;

            // Projective::print(groups[i]);
        }
        return 0;
    }

    template <typename Projective>
    static unsigned calc_groups_sum(Projective *result, Projective *groups, unsigned groups_num, unsigned *slice_bit)
    {
        Projective group_sum = {0x0};
        Projective g = {0x0};
        Projective g0 = groups[0];
        Projective tmp = {0x0};

        Projective::init(group_sum);
        Projective::init(tmp);

        for (unsigned i = 0; i < (groups_num - 1); i++)
        {
            group_sum = group_sum + groups[groups_num - i - 1];

            for (unsigned j = 0; j < slice_bit[groups_num - i - 1 - 1]; j++)
            {
                group_sum = group_sum + group_sum;
            }
        }

        *result = group_sum + g0;

        return 0;
    }

    template <typename Fr>
    static void get_slice_bit(unsigned *slice_bit, unsigned groups_num, unsigned bit_s)
    {
        for (int j = 0; j < groups_num; j++)
        {
            unsigned index_start = j * bit_s;

            unsigned index_end = (j < (groups_num - 1)) ? ((j + 1) * bit_s - 1) : (Fr::BC - 1);

            slice_bit[j] = index_end - index_start + 1;
        }
    }

    unsigned get_group_number_64bit(unsigned bit_s)
    {
        return get_group_number<CURVE_BLS12_377::fr_64bit>(bit_s);
    }

    void get_slice_bit_64bit(unsigned *slice_bit, unsigned groups_num, unsigned bit_s)
    {
        get_slice_bit<CURVE_BLS12_377::fr_64bit>(slice_bit, groups_num, bit_s);
    }

    void calc_groups_sum_64bit(void *result, void *groups, unsigned groups_num, unsigned *slice_bit)
    {
        calc_groups_sum<CURVE_BLS12_377::projective_exed_64bit>(
            static_cast<CURVE_BLS12_377::projective_exed_64bit *>(result),
            static_cast<CURVE_BLS12_377::projective_exed_64bit *>(groups),
            groups_num, slice_bit);
    }

    void get_slices_info_64bit(unsigned groups_num, unsigned bit_s, vector<unsigned> &index_start, vector<unsigned> &index_end, vector<unsigned> &m, vector<unsigned> &n, vector<unsigned> &m_shift, vector<unsigned> &n_shift)
    {
        get_slices_info<CURVE_BLS12_377::fr_64bit>(groups_num, bit_s, index_start, index_end, m, n, m_shift, n_shift);
    }

    void msm_batch_calc_group_post_process_64bit(void *groups, void *segments_results, unsigned groups_start, unsigned groups_end, unsigned buckets_num, unsigned *slice_bit)
    {
        msm_batch_calc_group_post_process<CURVE_BLS12_377::projective_exed_64bit>(
            static_cast<CURVE_BLS12_377::projective_exed_64bit *>(groups),
            static_cast<CURVE_BLS12_377::projective_exed_64bit *>(segments_results),
            groups_start, groups_end, buckets_num, HW_CALC_GROUP_SPLIT_SUB_GROUP_NUM_FOR_ZPRIZE_2023);
    }
}
