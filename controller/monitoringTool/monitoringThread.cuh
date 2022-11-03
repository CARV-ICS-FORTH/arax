#ifndef MONITORINGTHREAD
#define MONITORINGTHREAD
using namespace::std;

/*Struct to describe GPU info */
struct GPU_info
{
    unsigned int gpuId; 
    long     int timestamp;
    unsigned int utilGPU;
    unsigned int utilMemory;
    unsigned int power;
    char         gpuName[64];
};
/*Struct to describe CPU info*/
struct CPU_info
{
    unsigned int cpuId;
    long     int timestamp;	
    unsigned int utilCPU;
};
/*Struct to describe CPU energy consumption*/
struct CPU_energy
{
    long     int timestamp;
    uint64_t     energyCPU[2];
};


#endif
