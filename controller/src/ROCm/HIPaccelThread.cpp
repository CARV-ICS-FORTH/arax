#include "HIPaccelThread.h"
#include "core/arax_data.h"
#include "core/arax_data_private.h"
#include "definesEnable.h"
#include "hip/hip_runtime.h"
#include "hip_utils.h"
#include "prepareHIPGPU.h"
#include "utils/timer.h"
#include <iostream>
#include <set>

#include "Utilities.h"
#include <sstream>

int hip_alloc_gpu(arax_data_s *data);

int hip_free_gpu(arax_data_s *data);

static bool registered   = false;
static bool unregistered = false;

using namespace std;

unordered_map<accelThread *, hipStream_t> HIPaccelThread::a2s;
unordered_map<hipStream_t, accelThread *> HIPaccelThread::s2a;

HIPaccelThread::HIPaccelThread(arax_pipe_s *v_pipe, AccelConfig &conf)
    : accelThread(v_pipe, conf)
{
    try {
        jsonGetSafe(conf.conf, "pci_id", pciId, "device pci id");
    } catch (std::runtime_error &err) {
        throw std::runtime_error("While parsing " + conf.name + " -> "
                + err.what());
    }
}

HIPaccelThread::~HIPaccelThread(){ }

typedef arax_task_state_e (HipFunctor)(arax_task_msg_s *, hipStream_t *stream);

/*Executes an operation (syncTo, SyncFrom, free, kernel)*/
void HIPaccelThread::executeOperation(AraxFunctor *functor,
  arax_task_msg_s *                                arax_task)
{
    ((HipFunctor *) functor)(arax_task, &stream);
}

/*initializes the HIP accelerator*/
bool HIPaccelThread::acceleratorInit()
{
    /*Find the number of GPUs that exist in the current node*/
    static int GPUExistInSystem[128] = { 0 };

    /* Set Device for each accelThread; Need to be performed for evevery
     * thread-stream */
    if (setHIPDevice(pciId) == true) {
        cout << "HIP GPU initialization done: " << pciId << endl;
    } else {
        cout << "Failed to initialize HIP device." << endl;
        return false;
    }

    /*Operations that need to be performed once for all GPUs*/
    if (!__sync_bool_compare_and_swap(&GPUExistInSystem[pciId], 0, 1)) {
        /*Pin memory once for all the GPUs in the system */

        /*    if (!registered) {
         *    hipError_t hip_retval =
         *        hipHostRegister(v_pipe, v_pipe->shm_size, hipHostRegisterMapped);
         *    if (hip_retval != hipSuccess) {
         *      cerr << "Error" << hip_retval << endl;
         *      throw runtime_error(hipGetErrorString(hip_retval) +
         *                          string(" (hipHostRegister)"));
         *    }
         *    registered = true;
         *  }*/
        /*Get the total number of GPUs*/
        int numberOfGPUS = numberOfHipDevices();
        if (pciId > numberOfGPUS) {
            cout << "The HIP device with id -" << pciId << "- does not exist!!"
                 << endl;
            cout << "Please set a HIP device (second column in .config) with id "
                "smaller than "
                 << numberOfGPUS << endl;
            cout << "The system will exit..." << endl;
            return false;
        }
    }

    // Create one stream per accelThread
    if (hipStreamCreateWithFlags(&(this->stream), hipStreamNonBlocking) !=
      hipSuccess)
    {
        cout << "error creating stream\n";
        exit(1);
    } else {
        int device = -1;
        hipGetDevice(&device);
        cout << "Using stream :" << (void *) this->stream << " on device " << device
             << " (should be " << pciId << ").\n";
        a2s[this]         = this->stream;
        s2a[this->stream] = this;
    }

    /* Prepare every accelThread - Stream*/
    if (prepareHIPStream((getStream(this))) == false) {
        cout << "Failed to prepare HIP device " << endl;
        return false;
    }

    return true;
} // HIPaccelThread::acceleratorInit

/*Get total size from gpu*/
size_t HIPaccelThread::getAvailableSize()
{
    size_t available;
    size_t total;
    // hipMemGetInfo can not receive NULL as argument-->Segfault
    hipError_t err = hipMemGetInfo(&available, &total);

    if (err != hipSuccess) {
        throw runtime_error("hipMemGetInfo " + string(__func__) + "()!");
    }
    return available;
}

/*Get total size from gpu*/
size_t HIPaccelThread::getTotalSize()
{
    size_t total;
    size_t free;
    // hipMemGetInfo can not receive NULL as argument-->Segfault
    hipError_t err = hipMemGetInfo(&free, &total);

    if (err != hipSuccess) {
        throw runtime_error("hipMemGetInfo " + string(__func__) + "()!");
    }
    return total;
}

/*Releases the CPU accelerator*/
void HIPaccelThread::acceleratorRelease()
{
    static int GPUExistInSystem[128] = { 0 };

    if (__sync_bool_compare_and_swap(&GPUExistInSystem[pciId], 0, 1)) {
        if (!unregistered) {
            arax_clean();
        }
        unregistered = true;
    }
}

void HIPaccelThread::printOccupancy(){ }

hipStream_t HIPaccelThread::getStream(accelThread *thread)
{
    return a2s.at(thread);
}

accelThread * HIPaccelThread::getThread(hipStream_t stream)
{
    return s2a.at(stream);
}

int hip_free_gpu(arax_data_s *data)
{
    if (!data->remote) {
        return 1;
    }

    if (hipFree(data->remote) != hipSuccess) {
        cerr << "Hip Free FAILED for " << (void *) data << " "
             << (void *) data->remote << endl;
        return 0;
    }
    data->remote = 0;
    return 1;
}

// check for duplicate allocations in accelerator
int hip_alloc_gpu(arax_data_s *data)
{
    // printf("alloc data OP work or not: %lu\n",arax_data_size(data));
    hipError_t hip_err;

    if (data->remote) {
        cerr << "Duplicate " << __func__ << "()!\n";
        throw runtime_error("Duplicate " + string(__func__) + "()!");
    }

    if (arax_data_size(data) != 0) {
        hip_err = hipMalloc(&(data->remote), arax_data_size(data));
    } else {
        hip_err      = hipSuccess;
        data->remote = NULL;
    }

    if (hip_err != hipSuccess) {
        cerr << "hipMalloc FAILED " << (void *) data << endl;
        return 0;
    }
    return 1;
}

bool hip_Host2GPU(arax_task_msg_s *arax_task, vector<void *> &ioHD);

bool hip_Host2GPUAsync(arax_task_msg_s *arax_task, vector<void *> &ioHD,
  hipStream_t stream)
{
    return hip_Host2GPU(arax_task, ioHD);
}

bool hip_Host2GPU(arax_task_msg_s *arax_task, vector<void *> &ioHD)
{
    arax_data_s *data;
    bool completed = true;

    for (int arg = 0; arg < arax_task->in_count + arax_task->out_count; arg++) {
        data = (arax_data_s *) arax_task->io[arg];

        if (!data->remote) { // No CUDA allocation
                             /* Allocate memory to the device for all the inputs*/
            if (!hip_alloc_gpu(data)) {
                cerr << "hipMalloc FAILED for input: " << arg << endl;
                /* inform the producer that he has made a mistake*/
                arax_task->state = task_failed;
                completed        = false;
            }
        }

        if (arg < (arax_task->in_count)) {
            if (!(data->flags & REMT_SYNC)) {
                cerr << "For input [" << arg
                     << "]: " << ((arg < arax_task->in_count) ? "[In ]" : "[Out]")
                     << " remote " << (void *) data
                     << " not synced to_remote! (flag=" << data->flags << ")\n";
            }
        }

        ioHD.push_back(data->remote);
    }

    return completed;
}

bool hip_GPU2Host(arax_task_msg_s *arax_task, vector<void *> &ioDH);

bool hip_GPU2HostAsync(arax_task_msg_s *arax_task, vector<void *> &ioDH,
  hipStream_t stream)
{
    return hip_GPU2Host(arax_task, ioDH);
}

/* Hip Memcpy from Device to host*/
bool hip_GPU2Host(arax_task_msg_s *arax_task, vector<void *> &ioDH)
{
    std::cerr << "Using deprecated function " << __func__ << "() !\n ";

    arax_task_mark_done(arax_task, task_completed);

    return true;
}

/* Free Device memory */
bool hip_GPUMemFree(vector<void *> &io){ return true; }

/***************** MIGRATION functions ******************/
bool HIPaccelThread::alloc_no_throttle(arax_data_s *data)
{
    // data already allocated
    if (data->remote) {
        return true;
    }
    // std::cerr<<__FILE__<<" "<<__func__<<std::endl;
    size_t sz = arax_data_size(data);

    arax_assert(data->accel);
    arax_assert(((arax_vaccel_s *) (data->accel))->phys);
    hipError_t err = hipMalloc(&(data->remote), sz);

    HIP_ERROR_FATAL(err);

    return true;
}

// #define DEBUG_PRINTS

void HIPaccelThread::alloc_remote(arax_data_s *vdata)
{
    if (vdata->remote)
        return;

    size_t sz = arax_data_size(vdata);

    arax_assert(vdata->accel);
    arax_assert((arax_accel_s *) ((arax_vaccel_s *) (vdata->accel))->phys);
    vdata->phys = ((arax_accel_s *) ((arax_vaccel_s *) (vdata->accel))->phys);
    #ifdef DEBUG_PRINTS
    std::cerr << "AMD : " << __func__ << " data: " << vdata
              << " ,remote: " << vdata->remote << " size: " << sz << std::endl;
    #endif

    arax_accel_size_dec(((arax_vaccel_s *) (vdata->accel))->phys, sz);

    hipError_t err = hipMalloc(&(vdata->remote), sz);
    HIP_ERROR_FATAL(err);
    // Check data size in vt_MaxPoolForward we create buffers with 0 size
    if (sz != 0)
        arax_assert(vdata->remote);
}

void HIPaccelThread::sync_to_remote(arax_data_s *vdata)
{
    size_t sz = arax_data_size(vdata);

    #ifdef DEBUG_PRINTS
    std::cerr << "AMD : " << __func__ << " data: " << vdata
              << " ,remote: " << vdata->remote << std::endl;
    #endif
    hipError_t err;
    if (!alloc_no_throttle(vdata)) {
        std::cerr << __FILE__ << " " << __func__ << " Allocation failed. Exit"
                  << std::endl;
        abort();
    }
    err = hipMemcpyAsync(vdata->remote, arax_data_deref(vdata), sz,
        hipMemcpyHostToDevice, this->stream);
    HIP_ERROR_FATAL(err);
}

void HIPaccelThread::sync_from_remote(arax_data_s *vdata)
{
    hipError_t err;
    size_t sz = arax_data_size(vdata);

    if (vdata->remote) {
        #ifdef DEBUG_PRINTS
        std::cerr << "GPU : " << __func__ << " data: " << vdata
                  << " ,remote: " << vdata->remote << std::endl;
        #endif
        err = hipMemcpyAsync(arax_data_deref(vdata), vdata->remote, sz,
            hipMemcpyDeviceToHost, this->stream);
    }
    hipStreamSynchronize(this->stream);
    HIP_ERROR_FATAL(err);
}

void HIPaccelThread::free_remote(arax_data_s *vdata)
{
    if (vdata->remote) {
        #ifdef DEBUG_PRINTS
        std::cerr << "GPU : " << __func__ << " data: " << vdata
                  << " , remote: " << vdata->remote << std::endl;
        #endif
        hipError_t err = hipFree(vdata->remote);
        HIP_ERROR_FATAL(err);
        arax_assert((arax_accel_s *) ((arax_vaccel_s *) (vdata->accel))->phys);
        arax_accel_size_inc(((arax_vaccel_s *) (vdata->accel))->phys,
          arax_data_size(vdata));
        vdata->phys   = 0;
        vdata->remote = 0;
    }
}

USES_NOOP_RESET(HIPaccelThread)

REGISTER_ACCEL_THREAD(HIPaccelThread)
