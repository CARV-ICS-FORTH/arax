#include <stdio.h>
#include <iostream>
#include <fstream>
using namespace ::std;

int main() {
    ifstream fin;
    fin.open("/proc/cpuinfo", ios::in);

    ofstream fout;
    fout.open("new.txt", ios::out);
    char ch;
    char *model_name, *cpu_cores;
    char line[75];
    while (fin.get(ch)) 
    {
        fin.get(line, 75, '\n');
        model_name = strstr(line, "model name");
        cpu_cores = strstr(line, "cpu cores");

        if (model_name != NULL) 
        {
            fout << "Accelerator type is CPU \n" << model_name << endl;
        } else if (cpu_cores != NULL) {
            fout << cpu_cores << endl << "--------------------" << endl;
        }
    }
    // Number of CUDA devices
    int devCount;
    cudaGetDeviceCount(&devCount);
    if (devCount > 0) 
    {
        fout << "Accelerator type is NVIDIA GPU" << endl;
        fout << "Number of NVIDIA GPUs: " << devCount << endl;
        // Iterate through devices
        for (int i = 0; i < devCount; ++i) 
        {
            cudaDeviceProp devProp;
            cudaGetDeviceProperties(&devProp, i);
            fout << "model name       " << devProp.name << endl;
            fout << "Multi-Processors " << devProp.multiProcessorCount << endl;
            cudaSetDevice(i);
        }
    } else 
    {
        cout << "There is no CUDA device" << endl;
    }
    fin.close();
    return 0;
}
