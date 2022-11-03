#include "AraxLibUtilsGPU.h"
#include "arax.h"
#include "core/arax_data.h"
#include "core/arax_data_private.h"
#include "cuda_utils.h"
#include <atomic>
#include <condition_variable>
#include <sstream>

#define RED "\033[1;31m"
#define GREEN "\033[1;32m"
#define RESET "\033[0m"

bool print_once = true;
typedef arax_task_state_e(CudaFunctor)(arax_task_msg_s *, cudaStream_t *stream);

/* Enable SYNC or ASYNC in GPU transfers*/
//#define SYNC_H2D_TRANSFERS

//#define BREAKDOWNS_CONTROLLER
// define SYNC to take meassurments
#ifdef BREAKDOWNS_CONTROLLER
#define SYNC() deviceSynchronize()

void deviceSynchronize() {
  if (print_once) {
    cerr << " ========================================================="
         << std::endl;
    cerr << RED << "Async calls are not supported due to meassurments!!"
         << RESET << endl;
    cerr << "For performance disable breakdowns_controller in ccmake!" << endl;
    cerr << " ========================================================="
         << std::endl;
    print_once = false;
  }
  cudaDeviceSynchronize();
}
#else
#define SYNC()
#endif

/* Enable the same flag in caffe (src/caffe/syncedmem.cpp)
   to check that allocations match with free */
//#define DATA_LEAKS
#ifdef DATA_LEAKS
#define PRINT_DATA_LEAKS(data) printLeaks(data, __func__, __FILE__, __LINE__)
void printLeaks(arax_data_s *data, const char *func, const char *file,
                size_t line) {
  cerr << "Func: " << func << " ,Size: " << arax_data_size(data)
       << " ,vdata: " << data << " ,remote data " << data->remote << endl;
}
#else
#define PRINT_DATA_LEAKS(data)
#endif

//#define PRINT_SYNC_OR_ASYNC
#ifdef PRINT_SYNC_OR_ASYNC
#define PRINT_SYNC_ASYNC(color, type) printSyncAsync(color, type)
void printSyncAsync(const char *color, const char *type) {
  cerr << color << " " << type << RESET << endl;
}
#else
#define PRINT_SYNC_ASYNC(color, type)
#endif

//#define DEBUG_PRINTS_GPUMEMCPY
#ifdef DEBUG_PRINTS_GPUMEMCPY
#define PRINT_GPUMEMCPY(src, dst, args)                                        \
  printGPUMemcpy(src, dst, args, __func__, __FILE__, __LINE__)
void printGPUMemcpy(arax_data_s *src, arax_data_s *dst, memcpyArgs *args,
                    const char *func, const char *file, size_t line) {
  size_t sz_src = arax_data_size(src);
  size_t sz_dst = arax_data_size(dst);
  cerr << "Func: " << func << " ,file: " << file << " ,line: " << line
       << " ,src ptr: " << src << " ,src size: " << arax_data_size(src)
       << " ,dst ptr: " << dst << " ,dst size: " << arax_data_size(dst)
       << " ,src offset: " << args->src_offset
       << " ,dst offset: " << args->dst_offset
       << " ,(sz_src - src_offset): " << sz_src - args->src_offset
       << " ,(sz_dst - dst_offset): " << sz_dst - args->dst_offset
       << " , args->size: " << args->size << endl;
}
#else
#define PRINT_GPUMEMCPY(src, dst, args)
#endif

/*
 * Checks if \c data is aligned in GPU memory.
 */
void isAligned(arax_data_s *data) {
  if (((unsigned long)data->remote) % 128) {
    cerr << " Unaligned memory access: " << ((unsigned long)data->remote) % 128
         << endl;
    cerr << " ABORT!!! " << endl;
    abort();
  }
}

/*
 * Checks if a pointer is in GPU memory.
 */
void is_device_pointer(const void *ptr, int line) {
  cudaPointerAttributes attributes;
  cudaError_t err;
  err = cudaPointerGetAttributes(&attributes, ptr);
  CUDA_ERROR_FATAL(err);

  if (attributes.devicePointer != NULL) {
    // cerr<<attributes.memoryType<<" Device pointer\n";
    // cerr<<" Device pointer "<<ptr<< " line: "<<line<<endl;
  } else {
    cerr << " NO Device pointer " << ptr << " line: " << line << " ABORT...\n";
    abort();
  }
}

/*
 * Checks if \c data has been allocated in GPU, if not allocates.
 * \return True if already allocated or just allocated. False if allocation
 * failed.
 */
bool alloc_no_throttle(arax_data_s *data) {
  size_t sz = arax_data_size(data);
  // data already allocated
  if (data->remote) {
    return true;
  }
  arax_assert(data->accel);
  arax_assert((arax_accel_s *)((arax_vaccel_s *)(data->accel))->phys);
  data->phys = ((arax_accel_s *)((arax_vaccel_s *)(data->accel))->phys);
  arax_accel_size_dec(((arax_vaccel_s *)(data->accel))->phys,
                      arax_data_size(data));
  // isAligned(data);

  cudaError_t err = cudaMalloc(&(data->remote), sz);
  PRINT_DATA_LEAKS(data);
  CUDA_ERROR_FATAL(err);

  return true;
}
//#define DEBUG_PRINTS
/*Allocates data in the GPU*/
arax_task_state_e alloc_gpu_data(arax_task_msg_s *arax_task,
                                 cudaStream_t *stream) {
  arax_assert(arax_task->in_count == 0);
  arax_assert(arax_task->out_count == 1);
  arax_data_s *data = (arax_data_s *)arax_task->io[0];
  size_t sz = arax_data_size(data);

  // data already allocated
  if (data->remote) {
    // cerr<<"Have already remote. AllocData: "<<__FILE__<<__LINE__<<endl;
    arax_task_mark_done(arax_task, task_completed);
    return task_completed;
  }
  arax_assert((arax_accel_s *)((arax_vaccel_s *)(data->accel))->phys);
  // isAligned(data);
  cudaError_t err = cudaMalloc(&(data->remote), sz);
  CUDA_ERROR_FATAL(err);
#ifdef DEBUG_PRINTS
  std::cerr << __func__ << " task: " << arax_task << " data: " << data
            << " data remote: " << data->remote << " size: " << sz << std::endl;
#endif

  arax_assert(data->remote);
  data->phys = ((arax_accel_s *)((arax_vaccel_s *)(data->accel))->phys);

  arax_task_mark_done(arax_task, task_completed);
  return task_completed;
}
void arax_data_memcpy_gpu_cb(void *userData) {
  arax_task_msg_s *arax_task = (arax_task_msg_s *)userData;
  // arax_data_free(arax_task->io[0]);
  arax_task_free(arax_task);
}
/**
 * Performs a Remote to Remote copy
 */
arax_task_state_e gpu_memcpy(arax_task_msg_s *arax_task, cudaStream_t *stream) {
  memcpyArgs *args =
      (memcpyArgs *)arax_task_host_data(arax_task, sizeof(memcpyArgs));
  arax_data_s *src = (arax_data_s *)(arax_task->io[0]);
  arax_data_s *dst = (arax_data_s *)(arax_task->io[1]);
#ifdef DEBUG_PRINTS
  std::cerr << __func__ << " task: " << arax_task << " src: " << src
            << " offset: " << args->src_offset << " dst: " << dst
            << " offset: " << args->dst_offset << std::endl;
#endif

  size_t sz = args->size;
  if (args->size == 0) {
    std::cerr << __FILE__ << " " << __func__
              << " args->size is Zero. Please specify size! Abort.\n";
    abort();
  }

  cudaError_t err;
  size_t sz_src = arax_data_size(src);
  size_t sz_dst = arax_data_size(dst);

  PRINT_GPUMEMCPY(src, dst, args);

  if (!alloc_no_throttle(dst)) {
    arax_task_mark_done(arax_task, task_failed);
    return task_failed;
  }
  if (!alloc_no_throttle(src)) {
    arax_task_mark_done(arax_task, task_failed);
    return task_failed;
  }

  arax_assert(dst->remote);
  arax_assert(src->remote);

  if ((sz_dst - (args->dst_offset) < sz)) {
    cerr << __FILE__ << " " << __LINE__ << " " << __func__ << endl;
    cerr << "arax_data_size(dst)-dst_offset: " << sz_dst - args->dst_offset
         << " is < args->size: " << sz << endl;
    abort();
  }
  if ((sz_src - (args->src_offset)) < sz) {
    cerr << __FILE__ << " " << __LINE__ << " " << __func__ << endl;
    cerr << " arax_data_size(src)-src_offset: " << sz_src - (args->src_offset)
         << " is < args->size: " << sz << endl;
    abort();
  }

  char *src_ptr = ((char *)src->remote) + args->src_offset;
  char *dst_ptr = ((char *)dst->remote) + args->dst_offset;
  // Remote to remote copy
  err =
      cudaMemcpyAsync(dst_ptr, src_ptr, sz, cudaMemcpyDeviceToDevice, *stream);

#ifdef DEBUG
  float *src_h = 0, *dst_h = 0;
  src_h = (float *)malloc(sz);
  dst_h = (float *)malloc(sz);

  cudaMemcpy(src_h, ((char *)src->remote) + args->src_offset, sz,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(dst_h, ((char *)dst->remote) + args->dst_offset, sz,
             cudaMemcpyDeviceToHost);

  std::cerr << __func__ << " AFTER task: " << arax_task << " dst: " << dst
            << " dst value: " << src_h[0] << " src: " << src
            << " src value: " << dst_h[0] << std::endl;
#endif

  CUDA_ERROR_FATAL(err);

  // If arax_task is async free task from controller
  if (args->sync == false) {
    err = cudaStreamSynchronize(*stream);
    CUDA_ERROR_FATAL(err);
    return task_completed;
  }
  arax_task_mark_done(arax_task, task_completed);
  return task_completed;
}

/**
 * Performs a Memset
 */
void arax_data_memset_gpu_cb(void *userData) {
  arax_task_msg_s *arax_task = (arax_task_msg_s *)userData;
  arax_task_mark_done(arax_task, task_completed);
  arax_task_free(arax_task);
}
arax_task_state_e gpu_memset(arax_task_msg_s *arax_task, cudaStream_t *stream) {
  cudaError_t err;
  memsetArgs *args =
      (memsetArgs *)arax_task_host_data(arax_task, sizeof(memsetArgs));
  arax_data_s *data = (arax_data_s *)(arax_task->io[0]);

  if (!alloc_no_throttle(data)) {
    arax_task_mark_done(arax_task, task_failed);
    return task_failed;
  }
  arax_assert(data->remote);
  char *dst = ((char *)data->remote) + args->data_offset;
  err = cudaMemsetAsync((void *)dst, args->value, args->size, *stream);
#ifdef DEBUG_PRINTS
  std::cerr << __func__ << " task: " << arax_task << " data: " << data
            << " remote: " << data->remote << " offset: " << args->data_offset
            << std::endl;
#endif

  CUDA_ERROR_FATAL(err);
  arax_task_mark_done(arax_task, task_completed);
  return task_completed;
}

arax_task_state_e gpu_memfree(arax_task_msg_s *arax_task,
                              cudaStream_t *stream) {
  void **args = (void **)arax_task_host_data(arax_task, sizeof(void *) * 4);
  void *ptrAtDevice = args[1];
  size_t size = (size_t)args[2];
  arax_vaccel_s *accel_ptr = (arax_vaccel_s *)args[3];
#ifdef DEBUG_PRINTS
  void *host = args[0];
  std::cerr << __func__ << " task: " << arax_task << " data: " << host
            << " remote: " << ptrAtDevice << " VAQ phys: " << accel_ptr
            << std::endl;
#endif
  cudaError_t err = cudaFree(ptrAtDevice);
  arax_assert(accel_ptr);
  // increment data->phys
  arax_accel_size_inc(accel_ptr, size);

  arax_task_mark_done(arax_task,
                      (err != cudaSuccess) ? task_failed : task_completed);

  CUDA_ERROR_FATAL(err);

  arax_task_free(arax_task);

  return task_completed;
}

arax_task_state_e init_phys(arax_task_msg_s *arax_task, cudaStream_t *stream) {
  cudaStreamSynchronize(*stream);
  arax_task_mark_done(arax_task, task_completed);
  return task_completed;
}

void arax_data_set_gpu_cb(void *userData) {
  arax_task_msg_s *arax_task = (arax_task_msg_s *)userData;
  arax_task_mark_done(arax_task, task_completed);
  arax_task_free(arax_task);
}

arax_task_state_e arax_data_set_gpu(arax_task_msg_s *arax_task,
                                    cudaStream_t *stream) {
  cudaError_t err;
  arax_assert(arax_task->in_count == 0);
  arax_assert(arax_task->out_count == 1);
  void *host_src = arax_task_host_data(arax_task, arax_task->host_size);
  arax_data_s *data = (arax_data_s *)(arax_task->io[0]);
  if (!alloc_no_throttle(data)) {
    arax_task_mark_done(arax_task, task_failed);
    return task_failed;
  }
#ifdef DEBUG_PRINTS
  std::cerr << "H2D Size: " << arax_data_size(data) << std::endl;
  std::cerr << __func__ << " task: " << arax_task << " data: " << data
            << " remote: " << data->remote << " data: " << data
            << " size: " << arax_task->host_size << " host: " << host_src
            << std::endl;
#endif
//#define BREAKDOWNS
#ifdef BREAKDOWNS
  auto start_1 = std::chrono::high_resolution_clock::now();
#endif
  arax_assert(data->remote);
  err = cudaMemcpyAsync(data->remote, host_src, arax_task->host_size,
                        cudaMemcpyDefault, *stream);
  CUDA_ERROR_FATAL(err);
  cudaStreamSynchronize(*stream);
#ifdef BREAKDOWNS
  auto end_1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_milli1 = end_1 - start_1;
  std::cerr << "H2D time : " << elapsed_milli1.count() << " ms" << std::endl;
#endif
  arax_task_free(arax_task);
  return task_completed;
}

void arax_data_get_gpu_cb(void *userData) {
  arax_task_msg_s *arax_task = (arax_task_msg_s *)userData;
  arax_task_mark_done(arax_task, task_completed);
}

arax_task_state_e arax_data_get_gpu(arax_task_msg_s *arax_task,
                                    cudaStream_t *stream) {
  cudaError_t err;
  arax_assert(arax_task->in_count == 0);
  arax_assert(arax_task->out_count == 1);
  void *host_src = arax_task_host_data(arax_task, arax_task->host_size);
  arax_data_s *data = (arax_data_s *)(arax_task->io[0]);
  arax_assert(data->remote);
  err = cudaMemcpyAsync(host_src, data->remote, arax_data_size(data),
                        cudaMemcpyDefault, *stream);

  CUDA_ERROR_FATAL(err);
#ifdef DEBUG_PRINTS
  std::cerr << "D2H Size: " << arax_data_size(data) << std::endl;
  std::cerr << __func__ << " task: " << arax_task << " data: " << data
            << " remote: " << data->remote << std::endl;
#endif
  cudaStreamSynchronize(*stream);
  arax_task_mark_done(arax_task, task_completed);
  return task_completed;
}

ARAX_PROC_LIST_START()
ARAX_PROCEDURE("alloc_data", GPU, (AraxFunctor *)alloc_gpu_data, 0)
ARAX_PROCEDURE("free", GPU, (AraxFunctor *)gpu_memfree, 0)
ARAX_PROCEDURE("memset", GPU, (AraxFunctor *)gpu_memset, sizeof(memsetArgs))
ARAX_PROCEDURE("memcpy", GPU, (AraxFunctor *)gpu_memcpy, sizeof(memcpyArgs))
ARAX_PROCEDURE("init_phys", GPU, (AraxFunctor *)init_phys, 0)
ARAX_PROCEDURE("arax_data_set", GPU, (AraxFunctor *)arax_data_set_gpu, 0)
ARAX_PROCEDURE("arax_data_get", GPU, (AraxFunctor *)arax_data_get_gpu, 0)
ARAX_PROC_LIST_END()
