#include "CUDAaccelThread.h"
#include "AraxLibUtilsGPU.h"
#include "core/arax_data.h"
#include "core/arax_data_private.h"
#include "cuda_utils.h"
#include "definesEnable.h"
#include "prepareGPU.cuh"
#include "utils/timer.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <set>
#ifdef SM_OCCUPANCY
#include <occupancy.cuh>
#endif
#include "Utilities.h"
#include <sstream>

int alloc_gpu(arax_data_s *data);

int free_gpu(arax_data_s *data);

static bool registered   = false;
static bool unregistered = false;

using namespace std;

unordered_map<accelThread *, cudaStream_t> GPUaccelThread::a2s;
unordered_map<cudaStream_t, accelThread *> GPUaccelThread::s2a;

GPUaccelThread::GPUaccelThread(arax_pipe_s *v_pipe, AccelConfig &conf)
    : accelThread(v_pipe, conf)
{
    try {
        jsonGetSafe(conf.conf, "pci_id", pciId, "device pci id");
        jsonGetSafeOptional(conf.conf, "host_register", host_register,
          "host register", true);
        if (pciId < 0 || pciId > numberOfCudaDevices())
            throw std::runtime_error("Invalid pciID");
    } catch (std::runtime_error &err) {
        throw std::runtime_error("While parsing " + conf.name + " -> "
                + err.what());
    }
    start();
}

GPUaccelThread::~GPUaccelThread(){ }

typedef arax_task_state_e (CudaFunctor)(arax_task_msg_s *, cudaStream_t *stream);

/*Executes an operation (syncTo, SyncFrom, free, kernel)*/
void GPUaccelThread::executeOperation(AraxFunctor *functor,
  arax_task_msg_s *                                arax_task)
{
    ((CudaFunctor *) functor)(arax_task, &stream);
}

/*initializes the GPU accelerator*/
bool GPUaccelThread::acceleratorInit()
{
    /*Find the number of GPUs that exist in the current node*/
    static int GPUExistInSystem[128] = { 0 };

    /* Set Device for each accelThread; Need to be performed for every
     * thread-stream */
    if (setCUDADevice(pciId) == true) {
        cout << "GPU initialization done: " << pciId << endl;
    } else {
        cout << "Failed to initialize device." << endl;
        return false;
    }
    /*Operations that need to be performed once for all GPUs*/
    if (__sync_bool_compare_and_swap(&GPUExistInSystem[pciId], 0, 1)) {
        /*Pin memory once for all the GPUs in the system */
        if (!registered && host_register) {
            cudaError_t cuda_retval =
              cudaHostRegister(v_pipe, v_pipe->shm_size, cudaHostRegisterMapped);
            if (cuda_retval != cudaSuccess) {
                cerr << "Error" << cuda_retval << endl;
                throw runtime_error(cudaGetErrorString(cuda_retval)
                        + string(" (cudaHostRegister)"));
            }
            registered = true;
        }
    }

    // Create one stream per accelThread
    if (cudaStreamCreateWithFlags(&(this->stream), cudaStreamNonBlocking) !=
      cudaSuccess)
    {
        cout << "error creating stream\n";
        exit(1);
    } else {
        int device = -1;
        cudaGetDevice(&device);
        cout << "Using stream :" << (void *) this->stream << " on device " << device
             << " (should be " << pciId << ").\n";
        a2s[this]         = this->stream;
        s2a[this->stream] = this;
    }

    /* Prepare every accelThread - Stream*/
    if (prepareCUDAStream((getStream(this))) == false) {
        cout << "Failed to prepare device " << endl;
        return false;
    }

    #ifdef SM_OCCUPANCY
    start_event_collection();
    #endif

    #ifdef SAMPLE_OCCUPANCY
    start_event_collection();
    start_sampling();
    #endif

    return true;
} // GPUaccelThread::acceleratorInit

/*Get total size from gpu*/
size_t GPUaccelThread::getAvailableSize()
{
    size_t available;
    cudaError_t err = cudaMemGetInfo(&available, NULL);

    if (err != cudaSuccess) {
        throw runtime_error("cudaMemGetInfo " + string(__func__) + "()!");
    }
    return available;
}

/*Get total size from gpu*/
size_t GPUaccelThread::getTotalSize()
{
    size_t total;
    cudaError_t err = cudaMemGetInfo(NULL, &total);

    if (err != cudaSuccess) {
        throw runtime_error("cudaMemGetInfo " + string(__func__) + "()!");
    }
    return total;
}

/*Releases the CPU accelerator*/
void GPUaccelThread::acceleratorRelease()
{
    static int GPUExistInSystem[128] = { 0 };

    if (__sync_bool_compare_and_swap(&GPUExistInSystem[pciId], 0, 1)) {
        /*Pin memory once for all the GPUs in the system */
        if (!unregistered) {
            arax_clean();
        }
        unregistered = true;
    }
    #ifdef SAMPLE_OCCUPANCY
    stop_sampling();
    #endif
}

void GPUaccelThread::printOccupancy()
{
    #ifdef SM_OCCUPANCY
    get_occupancy();
    #endif
}

cudaStream_t GPUaccelThread::getStream(accelThread *thread)
{
    return a2s.at(thread);
}

accelThread * GPUaccelThread::getThread(cudaStream_t stream)
{
    return s2a.at(stream);
}

int free_gpu(arax_data_s *data)
{
    if (!data->remote) {
        return 1;
    }

    if (cudaFree(data->remote) != cudaSuccess) {
        cerr << "Cuda Free FAILED for " << (void *) data << " "
             << (void *) data->remote << endl;
        return 0;
    }
    data->remote = 0;
    return 1;
}

// check for duplicate allocations in accelerator
int alloc_gpu(arax_data_s *data)
{
    // printf("alloc data OP work or not: %lu\n",arax_data_size(data));
    cudaError_t cu_err;

    if (data->remote) {
        cerr << "Duplicate " << __func__ << "()!\n";
        throw runtime_error("Duplicate " + string(__func__) + "()!");
    }

    if (arax_data_size(data) != 0) {
        cu_err = cudaMalloc(&(data->remote), arax_data_size(data));
    } else {
        cu_err       = cudaSuccess;
        data->remote = NULL;
    }

    if (cu_err != cudaSuccess) {
        cerr << "cudaMalloc FAILED " << (void *) data << endl;
        return 0;
    }
    return 1;
}

bool Host2GPU(arax_task_msg_s *arax_task, vector<void *> &ioHD);

bool Host2GPUAsync(arax_task_msg_s *arax_task, vector<void *> &ioHD,
  cudaStream_t stream)
{
    return Host2GPU(arax_task, ioHD);
}

bool Host2GPU(arax_task_msg_s *arax_task, vector<void *> &ioHD)
{
    arax_data_s *data;
    bool completed = true;

    for (int arg = 0; arg < arax_task->in_count + arax_task->out_count; arg++) {
        data = (arax_data_s *) arax_task->io[arg];

        if (!data->remote) { // No CUDA allocation
                             // cerr<<"Allocate data from "<<__func__<<" in
                             // "<<__FILE__<<endl;
            /* Allocate memory to the device for all the inputs*/
            if (!alloc_gpu(data)) {
                cerr << "cudaMalloc FAILED for input: " << arg << endl;
                /* inform the producer that he has made a mistake*/
                arax_task->state = task_failed;
                completed        = false;
            }
        }

        /*    if (arg < (arax_task->in_count)) {
         *    if (!(data->flags & REMT_SYNC))
         *      cerr << "For input [" << arg
         *           << "]: " << ((arg < arax_task->in_count) ? "[In ]" : "[Out]")
         *           << " remote " << (void *)data
         *           << " not synced to_remote! (flag=" << data->flags << ")\n";
         *  }
         */
        ioHD.push_back(data->remote);
    }

    return completed;
}

bool GPU2Host(arax_task_msg_s *arax_task, vector<void *> &ioDH);

bool GPU2HostAsync(arax_task_msg_s *arax_task, vector<void *> &ioDH,
  cudaStream_t stream)
{
    return GPU2Host(arax_task, ioDH);
}

/* Cuda Memcpy from Device to host*/
bool GPU2Host(arax_task_msg_s *arax_task, vector<void *> &ioDH)
{
    std::cerr << "Using deprecated function " << __func__ << "() !\n ";

    arax_task_mark_done(arax_task, task_completed);

    return true;
}

/* Free Device memory */
bool GPUMemFree(vector<void *> &io){ return true; }

/***************** MIGRATION functions ******************/
bool GPUaccelThread::alloc_no_throttle(arax_data_s *data)
{
    // data already allocated
    if (data->remote) {
        return true;
    }
    // std::cerr<<__FILE__<<" "<<__func__<<std::endl;
    size_t sz = arax_data_size(data);

    arax_assert(data->accel);
    arax_assert(((arax_vaccel_s *) (data->accel))->phys);
    cudaError_t err = cudaMalloc(&(data->remote), sz);

    CUDA_ERROR_FATAL(err);

    return true;
}

// #define DEBUG_PRINTS
void GPUaccelThread::alloc_remote(arax_data_s *vdata)
{
    if (vdata->remote)
        return;

    size_t sz = arax_data_size(vdata);

    arax_assert(vdata->accel);
    arax_assert((arax_accel_s *) ((arax_vaccel_s *) (vdata->accel))->phys);
    vdata->phys = ((arax_accel_s *) ((arax_vaccel_s *) (vdata->accel))->phys);

    #ifdef DEBUG_PRINTS
    std::cerr << "GPU : " << __func__ << " data: " << vdata
              << " ,remote: " << vdata->remote << " size: " << sz << std::endl;
    #endif
    arax_accel_size_dec(((arax_vaccel_s *) (vdata->accel))->phys,
      arax_data_size(vdata));
    #ifdef BREAKDOWNS
    auto start_1 = std::chrono::high_resolution_clock::now();
    #endif

    cudaError_t err = cudaMalloc(&(vdata->remote), sz);
    CUDA_ERROR_FATAL(err);

    #ifdef BREAKDOWNS
    auto end_1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_milli1 = end_1 - start_1;
    std::cerr << "Data size: " << sz
              << " alloc_rmt time : " << elapsed_milli1.count() << " ms"
              << " th: " << this << std::endl;
    #endif

    // Check data size in vt_MaxPoolForward we create buffers with 0 size
    if (sz != 0)
        arax_assert(vdata->remote);
} // GPUaccelThread::alloc_remote

// #define BREAKDOWNS
void GPUaccelThread::sync_to_remote(arax_data_s *vdata)
{
    size_t sz = arax_data_size(vdata);
    cudaError_t err;

    #ifdef DEBUG_PRINTS
    std::cerr << "GPU : " << __func__ << " data: " << vdata
              << " ,remote: " << vdata->remote << std::endl;
    #endif
    #ifdef BREAKDOWNS
    auto start_1 = std::chrono::high_resolution_clock::now();
    #endif

    err = cudaMemcpyAsync(vdata->remote, arax_data_deref(vdata), sz,
        cudaMemcpyHostToDevice, this->stream);
    // err = cudaMemcpy(vdata->remote, arax_data_deref(vdata), sz,
    //                  cudaMemcpyHostToDevice);

    #ifdef BREAKDOWNS
    auto end_1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_milli1 = end_1 - start_1;
    std::cerr << "Data size: " << sz
              << " sync_to time : " << elapsed_milli1.count() << " ms"
              << " th: " << this << std::endl;
    #endif

    CUDA_ERROR_FATAL(err);
}

void GPUaccelThread::sync_from_remote(arax_data_s *vdata)
{
    cudaError_t err;
    size_t sz = arax_data_size(vdata);

    // If there is remote copy data. If data has never been moved to an
    // accelerator they do not have remote so do not copy!
    if (vdata->remote) {
        #ifdef DEBUG_PRINTS
        std::cerr << "GPU : " << __func__ << " data: " << vdata
                  << " ,remote: " << vdata->remote << std::endl;
        #endif
        #ifdef BREAKDOWNS
        auto start_1 = std::chrono::high_resolution_clock::now();
        #endif

        err = cudaMemcpyAsync(arax_data_deref(vdata), vdata->remote, sz,
            cudaMemcpyDeviceToHost, this->stream);
        // err = cudaMemcpy(arax_data_deref(vdata), vdata->remote, sz,
        //                  cudaMemcpyDeviceToHost);

        #ifdef BREAKDOWNS
        auto end_1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_milli1 = end_1 - start_1;
        std::cerr << "Data size: " << sz
                  << " sync_from time : " << elapsed_milli1.count() << " ms"
                  << " th: " << this << std::endl;
        #endif
    }
    CUDA_ERROR_FATAL(err);
}

void GPUaccelThread::free_remote(arax_data_s *vdata)
{
    if (vdata->remote) {
        #ifdef DEBUG_PRINTS
        std::cerr << "GPU : " << __func__ << " data: " << vdata
                  << " , remote: " << vdata->remote << std::endl;
        #endif
        #ifdef BREAKDOWNS
        auto start_1 = std::chrono::high_resolution_clock::now();
        #endif

        cudaError_t err = cudaFree(vdata->remote);
        CUDA_ERROR_FATAL(err);
        arax_assert((arax_accel_s *) ((arax_vaccel_s *) (vdata->accel))->phys);
        arax_accel_size_inc(((arax_vaccel_s *) (vdata->accel))->phys,
          arax_data_size(vdata));
        vdata->phys   = 0;
        vdata->remote = 0;
        #ifdef BREAKDOWNS
        auto end_1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_milli1 = end_1 - start_1;
        std::cerr << " free_remote time : " << elapsed_milli1.count() << " ms"
                  << " th: " << this << std::endl;
        #endif
    }
}

USES_NOOP_RESET(GPUaccelThread)

REGISTER_ACCEL_THREAD(GPUaccelThread)
