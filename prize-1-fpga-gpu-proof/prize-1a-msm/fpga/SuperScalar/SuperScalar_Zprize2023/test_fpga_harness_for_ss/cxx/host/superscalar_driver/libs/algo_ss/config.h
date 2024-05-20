#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <math.h>

// msm
#define BIT_S 16

// 1: scalar datas come from serialize_unchecked of arkwork
#define SCALAR_DATA_FROM_ARK_SERIALIZE      1

#define HW_CALC_GROUP_SPLIT 1
#define HW_CALC_GROUP_SPLIT_FOR_ZPRIZE_2023 1

#define HW_CALC_GROUP_SPLIT_SUB_GROUP_NUM_FOR_ZPRIZE_2023    128  // 128
#define HW_BUCKET_RANDOM_BASES_INDEX 0
#define HW_BUCKET_RANDOM_BASES_INDEX_DUMP 0
#define HW_BIT_FLAG_CONFICT 1

#define EXED_UNIFIED_PADD    1
#define EXED_UNIFIED_PADD_PRECOMPUTE    1
#define EXED_NONE_UNIFIED_PADD_PRECOMPUTE   1
#define SIGNED_BUCKET 1
#define HW_0_BUCKET 0
// Note
// You should consider about the number of the buckets in the last group if macro LAST_GROUP_BUCKET_REDUCE is equal to 1.
#define LAST_GROUP_BUCKET_REDUCE 0

#define LOAD_GOLDEN_INPUT_DATA_R_FLAG 0
#define FOR_GEN_TOOL_CASE 0

#define HW_BASES_DATA_CHANNEL_NUM 4
#define HW_BATCH_GROUPS_NUM 3
#define HW_BATCH_GROUPS_CHANNEL_NUM 4
#define HW_OUTPUT_RESULTS_CHANNEL_NUM 3

#endif
