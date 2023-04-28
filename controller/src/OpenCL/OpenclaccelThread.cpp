#include "OpenclaccelThread.h"
#include "core/arax_data.h"
#include "core/arax_data_private.h"
#include "definesEnable.h"
#include "utils/timer.h"
#include <iostream>

#include "Utilities.h"
#include "err_code.h"
#include "opencl_util.h"
#include <chrono>
#include <exception>
#include <fstream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>

using namespace std;
#define CL_ERR_PRINT                                                           \
    cerr << "OpenCL Error: " << err.what() << " returned "                       \
         << err_code(err.err()) << __FILE__ << " : " << __func__ << ":"          \
         << __LINE__ << endl;

mutex thread_mtx;

struct oclHandleStruct oclHandles;
cl_context defaultContext;
cl_context getDefaultContext(){ return defaultContext; }

unordered_map<accelThread *, cl_command_queue> a2s;
unordered_map<cl_command_queue, accelThread *> s2a;
std::map<string, cl_kernel> OpenCLaccelThread::kernels;

OpenCLaccelThread::OpenCLaccelThread(arax_pipe_s *v_pipe, AccelConfig &conf)
    : accelThread(v_pipe, conf)
{
    jsonGetSafe(conf.conf, "pci_id", pciId, "Integer pci id");

    picojson::array cl_list;
    picojson::array kernel_name_list;

    // Load all .cl files
    jsonGetSafe(conf.conf, "kn", kernel_name_list, "No \"kn\":[] defined");
    string krn_nm;

    for (auto name : kernel_name_list) {
        jsonCast(name, krn_nm);
        kernel_names.push_back(krn_nm);
    }

    jsonGetSafe(conf.conf, "bin", cl_list, "No \"bin\":[] defined");

    CL_file cfl;

    cfl.isBinary = true;
    for (auto cl_file : cl_list) {
        jsonCast(cl_file, cfl.file);
        kernel_files.push_back(cfl);
    }

    for (vector<CL_file>::iterator k_f = this->kernel_files.begin();
      k_f != this->kernel_files.end(); k_f++)
        cout << "Binary name: " << k_f->file.c_str() << std::endl;
}

OpenCLaccelThread::~OpenCLaccelThread(){ }

typedef arax_task_state_e (OpenclFunctor)(arax_task_msg_s *,
  cl_command_queue *stream);

/*Executes an operation (syncTo, SyncFrom, free, kernel)*/
void OpenCLaccelThread::executeOperation(AraxFunctor *functor,
  arax_task_msg_s *                                   arax_task)
{
    ((OpenclFunctor *) functor)(arax_task, &stream);
}

/*initializes the FPGA accelerator*/
bool OpenCLaccelThread::acceleratorInit()
{
    std::chrono::high_resolution_clock::time_point s_fpga_init;
    std::chrono::high_resolution_clock::time_point e_fpga_init;
    cl_int resultCL;

    thread_mtx.lock();
    static int GPUExistInSystem[128] = { 0 };

    if (__sync_bool_compare_and_swap(&GPUExistInSystem[pciId], 0, 1)) {
        cout << "OpenCL acceleratorInit() " << endl;
        cout << "=================================================" << endl << endl;
        // Find the number of GPUs that exist in the current node, saved in
        // this->numberOfDevices (and platforms in this->numberOfPlatforms)
        if (getNumberOfDevices() == false) {
            cout << "Failed to get devices " << endl;
            return false;
        }

        int DEVICE_ID_INUSED = pciId;
        size_t size;
        size_t sourceSize;

        cl_uint numPlatforms;
        cl_platform_id *targetPlatform = NULL;
        cl_uint deviceListSize;
        cl_context_properties cprops[3];

        oclHandles.context = NULL;
        oclHandles.devices = NULL;
        oclHandles.program = NULL;

        // Print devices
        display_device_info(&targetPlatform, &numPlatforms);
        // std::cerr<<"NumPlatforms: "<<numPlatforms<<std::endl;
        cl_device_type device_type = CL_DEVICE_TYPE_ACCELERATOR;
        validate_selection(targetPlatform, &numPlatforms, cprops, &device_type);
        // create an OpenCL context
        oclHandles.context =
          clCreateContextFromType(cprops, device_type, NULL, NULL, &resultCL);
        if ((resultCL != CL_SUCCESS) || (oclHandles.context == NULL)) {
            std::cerr << "clCreateContextFromType in " << __func__ << "failed!"
                      << std::endl;
            abort();
        }
        defaultContext = oclHandles.context;
        // Get the size of device list
        oclHandles.cl_status = clGetContextInfo(oclHandles.context,
            CL_CONTEXT_DEVICES, 0, NULL, &size);

        deviceListSize = (int) (size / sizeof(cl_device_id));
        if (oclHandles.cl_status != CL_SUCCESS) {
            // throw(string("exception in _clInit -> clGetContextInfo"));
            std::cerr << "clGetContextInfo in " << __func__ << "failed!" << std::endl;
            abort();
        }
        // Allocate the device list
        oclHandles.devices =
          (cl_device_id *) malloc(deviceListSize * sizeof(cl_device_id));
        if (oclHandles.devices == 0) {
            std::cerr << "Malloc failed in " << __func__ << "failed!" << std::endl;
            abort();
        }

        // Get the device list data
        oclHandles.cl_status = clGetContextInfo(
            oclHandles.context, CL_CONTEXT_DEVICES, size, oclHandles.devices, NULL);
        if (oclHandles.cl_status != CL_SUCCESS) {
            std::cerr << "clGetContextInfo failed in " << __func__ << "failed!"
                      << std::endl;
            abort();
        }
        s_fpga_init = std::chrono::high_resolution_clock::now();
        //  Iterate the kernel files created from the json
        for (vector<CL_file>::iterator k_f = this->kernel_files.begin();
          k_f != this->kernel_files.end(); k_f++)
        {
            // Import kernels from bitstream
            char *kernel_file_path = getVersionedKernelName(k_f->file.c_str());
            char *source = read_kernel(kernel_file_path, &sourceSize);
            oclHandles.program = clCreateProgramWithBinary(
                oclHandles.context, 1, oclHandles.devices, &sourceSize,
                (const unsigned char **) &source, NULL, &resultCL);
            if ((resultCL != CL_SUCCESS) || (oclHandles.program == NULL)) {
                std::cerr << "clCreateProgramWithBinary failed in " << __func__
                          << "failed!" << std::endl;
                abort();
            }
        }
    }
    for (int nKernel = 0; nKernel < kernel_names.size(); nKernel++) {
        /* get a kernel object handle for a kernel with the given name */
        cl_kernel kernel = clCreateKernel(
            oclHandles.program, (kernel_names[nKernel]).c_str(), &resultCL);
        std::cerr << "kernel: " << nKernel << " is : " << kernel_names[nKernel]
                  << std::endl;
        if ((resultCL != CL_SUCCESS) || (kernel == NULL)) {
            string errorMsg = "InitCL()::Error: Creating Kernel (clCreateKernel) \""
              + kernel_names[nKernel] + "\"";
            std::cerr << "clCreateKernel failed in " << __func__ << "failed!"
                      << std::endl;
            std::cerr << " Error msg: " << errorMsg << std::endl;
            abort();
        }
        this->kernels[(kernel_names[nKernel]).c_str()] = kernel;
    }

    e_fpga_init = std::chrono::high_resolution_clock::now();
    // Create an OpenCL command queue
    this->stream = clCreateCommandQueue(oclHandles.context, oclHandles.devices[0],
        0, &resultCL);
    std::cerr << "Thread: " << this << std::endl;
    if ((resultCL != CL_SUCCESS) || (this->stream == NULL)) {
        std::cerr << "clCreateCommandQueue failed in " << __func__ << "failed!"
                  << std::endl;
        abort();
    } else {
        cout << "Using stream :" << (void *) this->stream << " on PCIe " << pciId
             << std::endl;
        a2s[this]         = this->stream;
        s2a[this->stream] = this;
    }
    std::chrono::duration<double, std::milli> fpga_conf =
      e_fpga_init - s_fpga_init;

    std::cerr << "--> FPGA configuration time: " << std::fixed
              << fpga_conf.count() << " ms" << std::endl;

    thread_mtx.unlock();

    return true;
} // OpenCLaccelThread::acceleratorInit

void OpenCLaccelThread::acceleratorRelease()
{
    std::cerr << "Call Release resources!!" << std::endl;

    for (int nKernel = 0; nKernel < kernel_names.size(); nKernel++) {
        cl_int resultCL =
          clReleaseKernel(this->kernels[(kernel_names[nKernel]).c_str()]);
        if (resultCL != CL_SUCCESS) {
            std::cerr << "clReleaseKernel failed in " << __func__ << "failed!"
                      << std::endl;
            abort();
        }
        this->kernels.clear();
        resultCL = clReleaseProgram(oclHandles.program);
        if (resultCL != CL_SUCCESS) {
            std::cerr << "clReleaseProgram failed in " << __func__ << "failed!"
                      << std::endl;
            abort();
        }
        resultCL = clReleaseCommandQueue(this->stream);
        if (resultCL != CL_SUCCESS) {
            std::cerr << "clReleaseCommandQueue failed in " << __func__ << "failed!"
                      << std::endl;
            abort();
        }
        resultCL = clReleaseContext(oclHandles.context);
        if (resultCL != CL_SUCCESS) {
            std::cerr << "clReleaseContext failed in " << __func__ << "failed!"
                      << std::endl;
            abort();
        }
    }
}

bool OpenCLaccelThread::loadKernels(){ return true; }

bool OpenCLaccelThread::getNumberOfDevices(){ return true; }

bool OpenCLaccelThread::initDevice(int deviceNo){ return false; }

// Perform a test copy to the device
bool OpenCLaccelThread::prepareDevice(){ return true; }

/*Get FPGA memory total size*/
size_t OpenCLaccelThread::getTotalSize(){ return 4 * 1024 * 1024 * 1024UL; }

/*Get FPGA current memory size */
size_t OpenCLaccelThread::getAvailableSize()
{
    return 4 * 1024 * 1024 * 1024UL;
}

void OpenCLaccelThread::printOccupancy(){ }

cl_command_queue getStream(accelThread *thread){ return a2s.at(thread); }

accelThread* getThread(cl_command_queue stream){ return s2a.at(stream); }

/**
 * Transfer Function Implementations
 */
bool Host2OpenCL(arax_task_msg_s *arax_task, vector<void *> &ioHD)
{
    std::cerr << "Using deprecated function " << __func__ << "() !\n ";

    arax_task_mark_done(arax_task, task_completed);
    return true;
}

/* Opencl Memcpy from buffers to host kokkinos*/
bool OpenCL2Host(arax_task_msg_s *arax_task, vector<void *> &ioDH)
{
    std::cerr << "Using deprecated function " << __func__ << "() !\n ";

    arax_task_mark_done(arax_task, task_completed);

    return true;
}

/* Free Device memory */
bool OpenCLMemFree(vector<void *> &io){ return true; }

// bool shouldResetGpu (int device)
bool shouldResetOpenCL(){ return true; }

extern bool resetPolicy;

void OpenCLaccelThread::reset(accelThread *caller){ }

cl_kernel OpenCLGetKernel(string name, cl_command_queue *stream)
{
    OpenCLaccelThread *th = (OpenCLaccelThread *) getThread(*stream);

    if (th->kernels.count(name) == 0) {
        abort();
    }

    /*  std::cerr << "thread: " << th << " Name: " << name << " stream: " <<
     * stream
     *          << std::endl;*/
    return th->kernels.at(name);
}

/*Copies data to the FPGA*/
bool copy_to_buffer(arax_data_s *data, bool sync){ return true; }

/*Copies data from the FPGA*/
bool copy_from_buffer(arax_data_s *data, bool sync){ return true; }

bool free_opencl(void *data){ return true; }

/********     DONE: HELPER FUNCTIONs            *******/

/***************** MIGRATION functions ******************/
bool OpenCLaccelThread::alloc_no_throttle(arax_data_s *data)
{
    // data already allocated
    if (data->remote) {
        return true;
    }
    // std::cerr<<__FILE__<<" "<<__func__<<std::endl;
    size_t sz = arax_data_size(data);

    arax_assert(data->accel);
    arax_assert(((arax_vaccel_s *) (data->accel))->phys);
    bool err = _clMallocRW(defaultContext, sz, data);

    #ifdef DEBUG
    if (err == false)
        return false;

    #endif
    return true;
}

// #define DEBUG_PRINTS
void OpenCLaccelThread::alloc_remote(arax_data_s *vdata)
{
    if (vdata->remote)
        return;

    size_t sz = arax_data_size(vdata);

    arax_assert(vdata->accel);
    arax_assert((arax_accel_s *) ((arax_vaccel_s *) (vdata->accel))->phys);
    vdata->phys = ((arax_accel_s *) ((arax_vaccel_s *) (vdata->accel))->phys);

    #ifdef DEBUG_PRINTS
    std::cerr << "FPGA : " << __func__ << " data: " << vdata
              << " ,remote: " << vdata->remote << " size: " << sz << std::endl;
    #endif
    arax_accel_size_dec(((arax_vaccel_s *) (vdata->accel))->phys,
      arax_data_size(vdata));
    bool err = _clMallocRW(defaultContext, sz, vdata);
    #ifdef DEBUG
    if (err == false) {
        arax_task_mark_done(arax_task, task_failed);
        return task_failed;
    }
    #endif

    // Check data size in vt_MaxPoolForward we create buffers with 0 size
    if (sz != 0)
        arax_assert(vdata->remote);
}

void OpenCLaccelThread::sync_to_remote(arax_data_s *vdata)
{
    size_t sz = arax_data_size(vdata);

    if (!alloc_no_throttle(vdata)) {
        std::cerr << __FILE__ << " " << __func__ << " Allocation failed. Exit"
                  << std::endl;
        abort();
    }
    #ifdef DEBUG_PRINTS
    std::cerr << "FPGA : " << __func__ << " data: " << vdata
              << " ,remote: " << vdata->remote << std::endl;
    #endif

    bool err = _clMemcpyH2D(this->stream, vdata, sz, arax_data_deref(vdata));
    _clFinish(this->stream);
    CL_ERROR_FATAL(err);
}

void OpenCLaccelThread::sync_from_remote(arax_data_s *vdata)
{
    size_t sz = arax_data_size(vdata);
    bool err;

    // If there is remote copy data. If data has never been moved to an
    // accelerator they do not have remote so do not copy!
    if (vdata->remote) {
        #ifdef DEBUG_PRINTS
        std::cerr << "FPGA : " << __func__ << " data: " << vdata
                  << " ,remote: " << vdata->remote << std::endl;
        #endif
        err = _clMemcpyD2H(this->stream, vdata, sz, arax_data_deref(vdata));
    }
    CL_ERROR_FATAL(err);
    _clFinish(this->stream);
}

void OpenCLaccelThread::free_remote(arax_data_s *vdata)
{
    if (vdata->remote) {
        #ifdef DEBUG_PRINTS
        std::cerr << "FPGA : " << __func__ << " data: " << vdata
                  << " , remote: " << vdata->remote << std::endl;
        #endif
        _clFree(vdata->remote);

        arax_assert((arax_accel_s *) ((arax_vaccel_s *) (vdata->accel))->phys);
        arax_accel_size_inc(((arax_vaccel_s *) (vdata->accel))->phys,
          arax_data_size(vdata));
        vdata->phys   = 0;
        vdata->remote = 0;
    }
}

REGISTER_ACCEL_THREAD(OpenCLaccelThread)
