

#include <sys/time.h>  

#include <cstdint>

namespace common
{

  #define TIMECONVERTER  1000000  // 1s = 1000 000 us

  int dump_file(void *data, uint64_t len, char *filepath);
  int dump_file1(void *data, uint64_t len, char *filepath);
  int dump_groups_result_file(void *data, int group_num, uint64_t len, char *filepath);
}
