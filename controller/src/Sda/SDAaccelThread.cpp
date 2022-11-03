#include "SDAaccelThread.h"
#include "definesEnable.h"
#include "timers.h"
#include "xcl.h"
#include <chrono>
#include <ctime>
#include <iostream>
#include <set>
#include <sstream>

using namespace std;

SDAaccelThread::SDAaccelThread(arax_pipe_s *v_pipe, AccelConfig &conf)
    : accelThread(v_pipe, conf) {
  std::istringstream iss(conf.init_params);
  std::string kernel_name;
  iss >> prof_thread_cpus;
  if (!iss)
    throw runtime_error(
        "SDA accelerator incorrect arguments:Missing Profiling thread cpumask");
  iss >> vendor; // Xilinx
  if (!iss)
    throw runtime_error("SDA accelerator incorrect arguments:Missing vendor");
  iss >> dev_addr_str; // xilinx:adm-pcie-ku3:1ddr:3.0
  if (!iss)
    throw runtime_error(
        "SDA accelerator incorrect arguments:Missing device address");
  iss >> xclbin;
  if (!iss)
    throw runtime_error(
        "SDA accelerator incorrect arguments:Missing xclbin file");
  while (iss) {
    iss >> kernel_name;
    if (iss) {
      kernels[kernel_name] = 0; /* Will create it at init */
      std::cerr << "SDA kernel: " << kernel_name << std::endl;
    }
  }

  if (!kernels.size())
    throw runtime_error(
        "SDA accelerator incorrect arguments:Missing kernel string(s)");
}
SDAaccelThread::~SDAaccelThread() {}

/*initializes the SDA accelerator*/
bool SDAaccelThread::acceleratorInit() {
  /* Profiling thread will be spawned here, set its affinity */
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t),
                         prof_thread_cpus.getSet());
  world = xcl_world_single(CL_DEVICE_TYPE_ACCELERATOR, vendor.c_str(),
                           dev_addr_str.c_str());
  for (auto &krnl : kernels) {
    krnl.second = xcl_import_binary(world, xclbin.c_str(), krnl.first.c_str());
    cout << "Registered " << krnl.first << ": " << (void *)kernels[krnl.first]
         << std::endl;
  }
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t),
                         getAccelConfig().affinity.getSet());
  cout << "SDA initalization done." << endl;
  return true;
}
/*Releases the CPU accelerator*/
void SDAaccelThread::acceleratorRelease() {
  cl_int err;
  for (auto krnl : kernels) {
    cout << "Releasing " << krnl.first << ":";

    err = clReleaseKernel(krnl.second);

    if (err == CL_SUCCESS)
      cout << "Success\n";
    else
      cout << "Fail(" << (int)err << ")\n";
  }

  xcl_release_world(world);
  cout << "SDA released." << endl;
}

typedef void(SDAFunctor)(arax_task_msg_s *, xcl_world, cl_kernel);

void SDAaccelThread::executeHostCode(void *functor, arax_task_msg_s *task) {
  std::string kname = ((arax_object_s *)task->proc)->name;
  (*(SDAFunctor **)(functor))(task, world, kernels["krnl_" + kname]);
}

/**
 * TODO: UGLY HACK TO AVOID API change PLZ FIXME
 */
extern arax_pipe_s *vpipe_s;

bool Host2SDA(arax_task_msg_s *arax_task, vector<void *> &ioHD, xcl_world world,
              cl_kernel krnl) {
  void *tmpIn;
  bool completed = true;

  /*Map araxdata with sda data*/
  map<arax_data *, void *> araxData2SDA;

#ifdef DATA_TRANSFER
  cout << "Number of inputs: " << arax_task->in_count << endl;
  cout << "Number of outputs: " << arax_task->out_count << endl;
#endif
  int mallocIn;
  for (mallocIn = 0; mallocIn < arax_task->in_count; mallocIn++) {
    /* Iterate till the number of inputs*/
    if (((((arax_data_s *)arax_task->io[mallocIn].arax_data)->place) &
         (Both)) == HostOnly) {
      ioHD.push_back(arax_data_deref(arax_task->io[mallocIn].arax_data));
      continue;
    }

    tmpIn = xcl_malloc(world, CL_MEM_READ_ONLY,
                       arax_data_size(arax_task->io[mallocIn].arax_data));
    clSetKernelArg(krnl, mallocIn, sizeof(cl_mem), &tmpIn);
    /*map between araxdata and cuda alloced data*/
    araxData2SDA[arax_task->io[mallocIn].arax_data] = tmpIn;
  }

  int memCpyIn;
  for (memCpyIn = 0; memCpyIn < arax_task->in_count; memCpyIn++) {
    tmpIn = araxData2SDA[arax_task->io[memCpyIn].arax_data];

    /* Copy inputs to the device */
    xcl_memcpy_to_device(world, (cl_mem)tmpIn,
                         arax_data_deref(arax_task->io[memCpyIn].arax_data),
                         arax_data_size(arax_task->io[memCpyIn].arax_data));

#ifdef DATA_TRANSFER
    cout << "Size of input " << memCpyIn
         << " is: " << arax_data_size(arax_task->io[memCpyIn].arax_data)
         << endl;
#endif

    ioHD.push_back(tmpIn);
  }
  int out;
  void *tmpOut;

  /*Alocate memory for the outputs */
  for (out = mallocIn; out < arax_task->out_count + mallocIn; out++) {
    if (((arax_data_s *)(arax_task->io[out].arax_data))->flags & ARAXINPUT) {
      tmpOut = araxData2SDA[arax_task->io[out].arax_data];
    } else {
      tmpOut = xcl_malloc(world, CL_MEM_WRITE_ONLY,
                          arax_data_size(arax_task->io[out].arax_data));
    }
    clSetKernelArg(krnl, mallocIn, sizeof(cl_mem), &tmpOut);

    /*End cudaMalloc for outputs - Start Kernel Execution time*/
    ioHD.push_back(tmpOut);
  }

  return completed;
}

/* Cuda Memcpy from Device to host*/
bool SDA2Host(arax_task_msg_s *arax_task, vector<void *> &ioDH, xcl_world world,
              cl_kernel krnl) {
  int out;
  bool completed = true;

  for (out = arax_task->in_count;
       out < arax_task->out_count + arax_task->in_count; out++) {

#ifdef DATA_TRANSFER
    cout << "Size of output " << out
         << " is: " << arax_data_size(arax_task->io[out].arax_data) << endl;
#endif

    xcl_memcpy_from_device(world, arax_data_deref(arax_task->io[out].arax_data),
                           (cl_mem)ioDH[out],
                           arax_data_size(arax_task->io[out].arax_data));
    completed = true;
    if (out == arax_task->out_count + arax_task->in_count - 1) {
      arax_task_mark_done(arax_task, task_completed);
    }
  }
  return completed;
}

/* Free Device memory */
bool SDAMemFree(vector<void *> &io) {
  bool completed = true;
  set<void *> unique_set(io.begin(), io.end());
  for (set<void *>::iterator itr = unique_set.begin(); itr != unique_set.end();
       itr++) {
    clReleaseMemObject((cl_mem)*itr);
  }
  return completed;
}

USES_NOOP_RESET(GPUaccelThread)

REGISTER_ACCEL_THREAD(SDAaccelThread)
