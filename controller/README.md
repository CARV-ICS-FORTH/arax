# Folder layout

*   src - includes the implementation for the accelerator threads and Schedulers
*   monitoringTool - a tool that monitors GPU/CPU utilization-power consumption
*   nvrtc - ptx loader

# Building

*   First, build Arax according to its instructions.
*   Then create a directory build (mkidir build)
    *   Inside (cd build) build type ccmake3 ../
    *   Without modifing anything type c and then g
    *   Then do make
*   cmake automaticaly checks the existance of cuda, in both cases (with, without cuda) it creates one executable named as arax\_controller that supports both situations.

# Run

To start the accelerator service ./build/arax\_controller conf.json.
More configuration examples can be found in example\_configs directory.
The directory example\_configs contains the following configurations:

a. AMD GPU with 1xstream (amd\_1strm.json) and 2xstreams (amd\_2strms.json)

b. NVIDIA GPU with 1xstream (nvidia\_1strm.json) and 2xstreams (nvidia\_2strms.json)

c. CPU (cpu.json)

d. FPGA (fpga\_lavaMD.json)

e. FPGA-2xNVIDIA-AMD (fpga\_2nv\_amd\_lavaMD.json)

# Create your own configuration

1.  paths include the dynamic libraries with the kernels.
    "paths": \["path/to/the/.so"]
2.  groups describe accelerators with the same properties (it can be used for more advanced scheduling policies).
3.  name of the group in the conf.json it is 1GPU.
4.  accels field contains information (arch, name, cpu\_core, job\_pref, pci\_id, and ptx) about each accelerator.
5.  arch describes the type of accelerator or processing unit (CPU, GPU, FPGA)
6.  job\_pref describes the type of jobs that an accelerator thread can execute.
7.  pci\_id the PCIe bus that the accelerator is plugged-in.
8.  sched hold the scheduling policy
