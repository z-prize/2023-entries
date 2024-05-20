#ifndef __MSM_H__
#define __MSM_H__

#include <vector>

#include "bls12_377_config.h"
#include "configuration.h"

using namespace std;

namespace msm_cmodel
{
    #define BIT_S_MASK ((int)(1 << BIT_S) - 1) // 7bit ->  0b1111111 = 0x7F

    struct MSM_CONFIGURATION
    {
        void *bases;
        void *scalars;
        void *result;
        uint64_t msm_size;
        CONFIGURATION *configuration;
    };

    template <typename Fr, typename Affine, typename Projective>
    struct EXECUTION_CONFIGURATION
    {
        Affine *bases;
        Fr *scalars;
        Projective *results;
        uint64_t msm_size;
        CONFIGURATION *configuration;
    };

    unsigned get_bit_s();
    unsigned get_group_number_64bit(unsigned bit_s);
    void get_slice_bit_64bit(unsigned *slice_bit, unsigned groups_num, unsigned bit_s);
    void get_slices_info_64bit(unsigned groups_num, unsigned bit_s, vector<unsigned> &index_start, vector<unsigned> &index_end, vector<unsigned> &m, vector<unsigned> &n, vector<unsigned> &m_shift, vector<unsigned> &n_shift);

    // batch
    void msm_batch_calc_group_post_process_64bit(void *groups, void *segments_results, unsigned groups_start, unsigned groups_end, unsigned buckets_num, unsigned *slice_bit);
    void calc_groups_sum_64bit(void *result, void *groups, unsigned groups_num, unsigned *slice_bit);
}

#endif
