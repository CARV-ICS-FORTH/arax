#ifndef ARAXLIB_UTILS_MEM_OPS
#define ARAXLIB_UTILS_MEM_OPS
#include "arax_pipe.h"

typedef struct {
  void *data; // arax_data_s*
  size_t data_offset;
  int value;
  size_t size;
} memsetArgs;

typedef struct {
  arax_data *src; // source arax_data_s*
  size_t src_offset;
  arax_data *dst; // destination arax_data_s*
  size_t dst_offset;
  bool sync; // If true, client will wait for mark done.
  size_t size;
} memcpyArgs;

#endif
