//C++ libs
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <signal.h>
#include <sys/time.h>
#include <string>
#include <string.h>
#include <fstream>
#include <inttypes.h>

#include <stdlib.h>
#include <errno.h>
#include <dirent.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

//CPU libs
#include "rd_stats.h"
#include "common.h"
#include "count.h"

#ifdef __CUDACC__
//Cuda libs
#include <nvml.h>
#endif
//User defined libs
#include "monitoringThread.cuh"
//read info from file
#define STAT "/proc/stat"

using namespace std;

/*Struct wirth all GPU info*/
GPU_info *gpuInfo;
CPU_info *cpuInfo;
CPU_energy *cpuEnergy;

/*vector with info per GPU/CPU */
vector <GPU_info*> gpuResourceUtil;
vector <CPU_info*> cpuResourceUtil;
vector <CPU_energy*> cpuEnergy_vector;

volatile bool shouldExit = false;

//Current timestamp used from both vectors CPU, GPU
long int ms = 0;

//Mpstat definitions
unsigned long long uptime[3] = {0, 0, 0};
unsigned long long uptime0[3] = {0, 0, 0};
/*
 * Nb of processors on the machine.
 * A value of 2 means there are 2 processors (0 and 1).
 */

struct tm mp_tstamp[3];
struct stats_cpu *st_cpu[3];

void secsSince1970()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	cerr<<"Seconds since Jan. 1, 1970: "<<tv.tv_sec<<endl;
	cerr<<"USeconds since Jan. 1, 1970: "<<1000000 * tv.tv_sec + tv.tv_usec<<endl;
}

void my_gettime(struct timeval * tp,struct timezone *tz)
{
	static timeval start;
	gettimeofday(tp,tz);

	if(!start.tv_sec)
		start = *tp;
	tp->tv_sec -= start.tv_sec;
	tp->tv_usec -= start.tv_usec;
}

#define gettimeofday my_gettime

/*CPU functions*/
void write_stats(int curr, int dis);
void write_stats_core(int prev, int curr, int dis, char *prev_string, char *curr_string);
void write_cpu_stats(int dis, unsigned long long g_itv, int prev, int curr,
		char *prev_string, char *curr_string, int tab, int *next);
void write_plain_cpu_stats(int dis, unsigned long long g_itv, int prev, int curr,
		char *prev_string, char *curr_string);

/* Define the function to be called when ctrl-c (SIGINT) signal is sent to
 * process*/
void signal_callback_handler(int signum) 
{
	cerr << "Caught signal " << signum << endl;
	shouldExit = true;
}

/*Read enenergy consumption of CPUs*/
uint64_t getPackageEnergy(int package)
{
	ifstream ifs("/sys/class/powercap/intel-rapl/intel-rapl:"+std::to_string(package)+"/energy_uj");
	unsigned long long energy;
	ifs >> energy;
	return energy;
}

/*
 ***************************************************************************
 * Read CPU statistics.
 *
 * IN:
 * @st_cpu	Structure where stats will be saved.
 * @nbr		Total number of CPU (including cpu "all").
 *
 * OUT:
 * @st_cpu	Structure with statistics.
 ***************************************************************************
 */

void read_stat_cpu(struct stats_cpu *st_cpu, int nbr,  unsigned long long *uptime, unsigned long long *uptime0)
{
	FILE *fp;
	struct stats_cpu *st_cpu_i;
	struct stats_cpu sc;
	char line[8192];
	int proc_nb;

	if ((fp = fopen(STAT, "r")) == NULL) 
	{
		fprintf(stderr, ("Cannot open %s: %s\n"), STAT, strerror(errno));
		exit(2);
	}

	while (fgets(line, sizeof(line), fp) != NULL) 
	{

		if (!strncmp(line, "cpu ", 4)) 
		{
			/*
			 * All the fields don't necessarily exist,
			 * depending on the kernel version used.
			 */
			memset(st_cpu, 0, STATS_CPU_SIZE);

			/*
			 * Read the number of jiffies spent in the different modes
			 * (user, nice, etc.) among all proc. CPU usage is not reduced
			 * to one processor to avoid rounding problems.
			 */
			sscanf(line + 5, "%llu %llu %llu %llu %llu %llu %llu %llu %llu %llu",
					&st_cpu->cpu_user,
					&st_cpu->cpu_nice,
					&st_cpu->cpu_sys,
					&st_cpu->cpu_idle,
					&st_cpu->cpu_iowait,
					&st_cpu->cpu_hardirq,
					&st_cpu->cpu_softirq,
					&st_cpu->cpu_steal,
					&st_cpu->cpu_guest,
					&st_cpu->cpu_guest_nice);

		}

		else if (!strncmp(line, "cpu", 3)) 
		{
			if (nbr > 1) 
			{
				/* All the fields don't necessarily exist */
				memset(&sc, 0, STATS_CPU_SIZE);
				/*
				 * Read the number of jiffies spent in the different modes
				 * (user, nice, etc) for current proc.
				 * This is done only on SMP machines.
				 */
				sscanf(line + 3, "%d %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu",
						&proc_nb,
						&sc.cpu_user,
						&sc.cpu_nice,
						&sc.cpu_sys,
						&sc.cpu_idle,
						&sc.cpu_iowait,
						&sc.cpu_hardirq,
						&sc.cpu_softirq,
						&sc.cpu_steal,
						&sc.cpu_guest,
						&sc.cpu_guest_nice);

				if (proc_nb < (nbr - 1)) {
					st_cpu_i = st_cpu + proc_nb + 1;
					*st_cpu_i = sc;
				}
			}
		}
	}

	fclose(fp);
}

void monitoringCPUcores(int cpu_nr)
{
	static int curr = 0, dis = 1;
	struct stats_cpu;

	/* Dont buffer data if redirected to a pipe */
	setbuf(stdout, NULL);

	/* Get time */
	get_localtime(&(mp_tstamp[curr]), 0);

	/* Read uptime and CPU stats */
	if (cpu_nr > 1) {
		uptime0[curr] = 0;
		read_uptime(&(uptime0[curr]));
	}

	read_stat_cpu(st_cpu[curr], cpu_nr + 1, &(uptime[curr]), &(uptime0[curr]));
	write_stats(curr, dis);
	curr ^= 1;
}

/*
 ***************************************************************************
 * Print statistics.
 *
 * IN:
 * @curr	Position in array where statistics for current sample are.
 * @dis		TRUE if a header line must be printed.
 ***************************************************************************
 */
void write_stats(int curr, int dis)
{
	char cur_time[2][TIMESTAMP_LEN];

	/* Get previous timestamp */
	if (is_iso_time_fmt()) 
	{
		strftime(cur_time[!curr], sizeof(cur_time[!curr]), "%H:%M:%S", &mp_tstamp[!curr]);
	}
	else 
	{
		strftime(cur_time[!curr], sizeof(cur_time[!curr]), "%X", &(mp_tstamp[!curr]));
	}

	/* Get current timestamp */
	if (is_iso_time_fmt()) 
	{
		strftime(cur_time[curr], sizeof(cur_time[curr]), "%H:%M:%S", &mp_tstamp[curr]);
	}
	else 
	{
		strftime(cur_time[curr], sizeof(cur_time[curr]), "%X", &(mp_tstamp[curr]));
	}

	write_stats_core(!curr, curr, dis, cur_time[!curr], cur_time[curr]);
}


/*
 ***************************************************************************
 * Core function used to display statistics.
 *
 * IN:
 * @prev	Position in array where statistics used	as reference are.
 *		Stats used as reference may be the previous ones read, or
 *		the very first ones when calculating the average.
 * @curr	Position in array where statistics for current sample are.
 * @dis		TRUE if a header line must be printed.
 * @prev_string	String displayed at the beginning of a header line. This is
 * 		the timestamp of the previous sample, or "Average" when
 * 		displaying average stats.
 * @curr_string	String displayed at the beginning of current sample stats.
 * 		This is the timestamp of the current sample, or "Average"
 * 		when displaying average stats.
 ***************************************************************************
 */
void write_stats_core(int prev, int curr, int dis,
		char *prev_string, char *curr_string )
{
	struct stats_cpu *scc, *scp;
	unsigned long long g_itv;
	int cpu, tab = 4, next = FALSE;
	/* Compute time interval */
	g_itv = get_interval(uptime[prev], uptime[curr]);
	/* Print CPU stats */
	write_cpu_stats(dis, g_itv, prev, curr, prev_string, curr_string,
			tab, &next);
	/* Fix CPU counter values for every offline CPU */
	for (cpu = 1; cpu <= cpu_nr; cpu++) {

		scc = st_cpu[curr] + cpu;
		scp = st_cpu[prev] + cpu;

		if ((scc->cpu_user    + scc->cpu_nice + scc->cpu_sys   +
					scc->cpu_iowait  + scc->cpu_idle + scc->cpu_steal +
					scc->cpu_hardirq + scc->cpu_softirq) == 0) {
			/*
			 * Offline CPU found.
			 * Set current struct fields (which have been set to zero)
			 * to values from previous iteration. Hence their values won't
			 * jump from zero when the CPU comes back online.
			 */
			*scc = *scp;
		}
	}
}
/*
 ***************************************************************************
 * Display CPU statistics in plain or JSON format.
 *
 * IN:
 * @dis		TRUE if a header line must be printed.
 * @g_itv	Interval value in jiffies multiplied by the number of CPU.
 * @prev	Position in array where statistics used	as reference are.
 *		Stats used as reference may be the previous ones read, or
 *		the very first ones when calculating the average.
 * @curr	Position in array where current statistics will be saved.
 * @prev_string	String displayed at the beginning of a header line. This is
 * 		the timestamp of the previous sample, or "Average" when
 * 		displaying average stats.
 * @curr_string	String displayed at the beginning of current sample stats.
 * 		This is the timestamp of the current sample, or "Average"
 * 		when displaying average stats.
 * @tab		Number of tabs to print (JSON format only).
 * @next	TRUE is a previous activity has been displayed (JSON format
 * 		only).
 ***************************************************************************
 */
void write_cpu_stats(int dis, unsigned long long g_itv, int prev, int curr,
		char *prev_string, char *curr_string, int tab, int *next)
{
	write_plain_cpu_stats(dis, g_itv, prev, curr, prev_string, curr_string);
}

/*
 ***************************************************************************
 * Display CPU statistics in plain format.
 *
 * IN:
 * @dis		TRUE if a header line must be printed.
 * @g_itv	Interval value in jiffies multiplied by the number of CPU.
 * @prev	Position in array where statistics used	as reference are.
 *		Stats used as reference may be the previous ones read, or
 *		the very first ones when calculating the average.
 * @curr	Position in array where current statistics will be saved.
 * @prev_string	String displayed at the beginning of a header line. This is
 * 		the timestamp of the previous sample, or "Average" when
 * 		displaying average stats.
 * @curr_string	String displayed at the beginning of current sample stats.
 * 		This is the timestamp of the current sample, or "Average"
 * 		when displaying average stats.
 ***************************************************************************
 */
void write_plain_cpu_stats(int dis, unsigned long long g_itv, int prev, int curr,
		char *prev_string, char *curr_string)
{
	struct stats_cpu *scc, *scp;
	unsigned long long pc_itv;
	int cpu;
	double user, sys;
	//Struct for gettimeofday
	cpuEnergy = new CPU_energy;
	for (cpu = 1; cpu <= cpu_nr; cpu++) 
	{
		cpuInfo = new CPU_info;

		scc = st_cpu[curr] + cpu;
		scp = st_cpu[prev] + cpu;

		/* Recalculate itv for current proc */
		pc_itv = get_per_cpu_interval(scc, scp);

		user = ll_sp_value(scp->cpu_user, scc->cpu_user, pc_itv);
		sys = ll_sp_value(scp->cpu_sys, scc->cpu_sys, pc_itv);

		cpuInfo->timestamp = ms ; 

		cpuInfo->utilCPU = user + sys;
		cpuInfo->cpuId = cpu;


		cpuResourceUtil.push_back( cpuInfo );
	}
	cpuEnergy->energyCPU[0] = getPackageEnergy(0);
	cpuEnergy->energyCPU[1] = getPackageEnergy(1);
	cpuEnergy_vector.push_back (cpuEnergy);
	cpuEnergy->timestamp = ms ;
}
/*
 ***************************************************************************
 * Read machine uptime, independently of the number of processors.
 *
 * OUT:
 * @uptime	Uptime value in jiffies.
 ***************************************************************************
 */
void read_uptime(unsigned long long *uptime)
{
	FILE *fp;
	char line[128];
	unsigned long up_sec, up_cent;

	if ((fp = fopen(UPTIME, "r")) == NULL)
		return;

	if (fgets(line, sizeof(line), fp) == NULL) {
		fclose(fp);
		return;
	}

	sscanf(line, "%lu.%lu", &up_sec, &up_cent);
	*uptime = (unsigned long long) up_sec * HZ +
		(unsigned long long) up_cent * HZ / 100;
	fclose(fp);

}
/*
 ***************************************************************************
 * Allocate stats structures and cpu bitmap. Also do it for NUMA nodes
 * (although the machine may not be a NUMA one). Assume that the number of
 * nodes is lower or equal than that of CPU.
 *
 * IN:
 * @nr_cpus	Number of CPUs. This is the real number of available CPUs + 1
 * 		because we also have to allocate a structure for CPU 'all'.
 ***************************************************************************
 */
void salloc_mp_struct(int nr_cpus)
{
	/*salloc_mp_struct starts*/
	for (int i = 0; i < 3; i++) {
		/*Allocate stats structures for all cpu cores provided by user */
		if ((st_cpu[i] = (struct stats_cpu *) malloc(STATS_CPU_SIZE * (cpu_nr) )) == NULL) 
		{
			perror("malloc");
			exit(4);
		}
		memset(st_cpu[i], 0, STATS_CPU_SIZE * (cpu_nr) );
	}

}

int main (int argc, char* argv[])
{
	if (argc != 3)
	{
		cerr<<"Add sampling time in microseconds."<<endl;
		cerr<<"Add the number of cpus in the node."<<endl;
		return -1;
	}
	secsSince1970();
	/*time interval between samples*/  
	long sampleTimer =  atoi(argv[1]);
#ifdef ALLCPUs
	/* What is the highest processor number on this machine? */
	cpu_nr = get_cpu_nr(~0, TRUE);
#else
	/*get the cores num from keyboard*/
	cpu_nr = (int) atoi(argv[2]);
#endif
#ifdef __CUDACC__
	/*GPU information*/
	unsigned int i;
	nvmlReturn_t result;
	/*Counter for gpus without nvml*/
	int deviceCountCuda = 0;
	cudaError_t err;
	err = cudaGetDeviceCount(&deviceCountCuda);
	if (err != cudaSuccess)
	{
		cerr<<__FILE__<<" "<<cudaGetErrorString(err)<<endl;
		return -1; 
	}

	if (deviceCountCuda == 0)
	{
		cerr<<"NO GPU!!"<<endl;
		return -1;
	}

#endif
	//Struct for gettimeofday
	struct timeval tp, tp_now;
	gettimeofday(&tp, NULL);
#ifdef __CUDACC__
	if (deviceCountCuda != 0)
	{
		// First initialize NVML library  
		result = nvmlInit();
		if (NVML_SUCCESS != result)         
		{                 
			printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
			printf("Press ENTER to continue...\n");
			getchar();
			nvmlShutdown();
			return -1;         
		}        

		/*Retrieve the number of GPUs in the system*/
		/*
		  result = nvmlDeviceGetCount(&device_count);
		  if (NVML_SUCCESS != result)
		  { 
		      printf("Failed to query device count: %s\n", nvmlErrorString(result));
	    	      nvmlShutdown();
		      return -1;
		  }
		*/
	}
#endif
	/*Allocate structs for CPU info*/
	salloc_mp_struct(cpu_nr+1);
	get_HZ();

#ifdef __CUDACC__
	cout<<"GPUs            :  "<<deviceCountCuda<<endl;
#else
	cout<<"GPUs            :  "<<0<<endl;
#endif
	cout<<"CPU cores       :  "<<cpu_nr<<endl;
	cerr<<"Sample interval :  "<<sampleTimer<<endl;

	/*Signal to stop monitoring*/
	signal(SIGINT, signal_callback_handler);

	/*Get Utilization and Power for all GPUs*/
	cerr <<"Press Ctrl+c to stop monitoring."<<endl;
	//#define TIMER_in_ms

	while (!shouldExit)
	{
		/*get timestamp in milli secs*/
		gettimeofday(&tp_now, NULL);
#ifdef TIMER_in_ms
		//cout<<"Timer in ms"<<endl;
		ms = (tp_now.tv_sec * 1000000 + tp_now.tv_usec)/1000 ;
#else//timer in micro sec (us)
		//cout<<"Timer in us"<<endl;
		ms = (tp_now.tv_sec * 1000000 + tp_now.tv_usec) ;
#endif
#ifdef __CUDACC__
		if (deviceCountCuda != 0 )
		{
			/*iterate through all GPUs found*/
			for (i = 0; i < deviceCountCuda; i++)
			{
				/*Utilization*/
				nvmlUtilization_t util;
				/*PCI no*/
				nvmlPciInfo_t pci;
				/*GPU obj*/
				nvmlDevice_t device; 
				/*Power consumption*/
				unsigned int powerConsumption;
				/*GPUs name*/
				char name[64];
				
				/*Create a struct instance per GPU*/
				gpuInfo = new GPU_info;



				// Query for device handle to perform operations on a device
				result = nvmlDeviceGetHandleByIndex(i, &device);         
				if (NVML_SUCCESS != result)         
				{                 
					printf("Failed to get handle for device %i: %s\n", i, nvmlErrorString(result));                 
					nvmlShutdown();
					//	return -1;         
				}

				result = nvmlDeviceGetName(device, name, sizeof(name)/sizeof(name[0]));
				if (NVML_SUCCESS != result)
				{ 
					printf("Failed to get name of device %i: %s\n", i, nvmlErrorString(result));
					nvmlShutdown();
					return -1;
				}

				//Get PCI id for the specified device
				result = nvmlDeviceGetPciInfo(device, &pci);
				if (NVML_SUCCESS == result)
				{ 
					/*Get PCI id for a specific device*/ 
					gpuInfo->gpuId = pci.bus;	
				}
				else{
					printf("Failed to get pci info for device %i: %s\n", i, nvmlErrorString(result));
					nvmlShutdown();
					return -1;
				}

				//Get GPU utilization
				result = nvmlDeviceGetUtilizationRates(device, &util);        
				if (NVML_SUCCESS == result)
				{
					/*get GPU/Memory Utilization*/
					gpuInfo->utilGPU = util.gpu;
					gpuInfo->utilMemory = util.memory;
				}
				else{	
					gpuInfo->utilGPU = 999;
					gpuInfo->utilMemory= 999;
//					printf("Failed to get utilization info for device %i (%s): %s\n", i , name , nvmlErrorString(result));
				}
				//Get power consumption
				result = nvmlDeviceGetPowerUsage(device, &powerConsumption);        
				if (NVML_SUCCESS == result)
				{
					/*get power consumption in mili watt*/
					gpuInfo->power = powerConsumption;
				}
				else{
					gpuInfo->power = 999;
//					printf("Failed to get power consumption for device %i (%s): %s\n", i , name , nvmlErrorString(result));
				}

				/*pass current timestamp*/
				gpuInfo->timestamp = ms ; 

				/*get GPU name*/
				strcpy(gpuInfo->gpuName , name);

				/*push results per GPU*/
				gpuResourceUtil.push_back(gpuInfo);

			}
		}
#endif
		/***************End monitoring GPUs*****************/
		monitoringCPUcores(cpu_nr);

		//samplerTimer in microsecond
		usleep(sampleTimer);

		if (shouldExit)
		{
			cerr << "Ctr+c is pressed EXIT!" << endl;
			break;
		}

		std::vector<CPU_energy*>::iterator itr_cpu_energy;
		std::vector<GPU_info*>::iterator itr;
		std::vector<CPU_info*>::iterator itr_cpu;

		unsigned long long energy[2] = {cpuEnergy_vector[0]->energyCPU[0], cpuEnergy_vector[0]->energyCPU[1]};
		/*CPUs + GPUS*/
#ifdef __CUDACC__
		if (deviceCountCuda != 0 )
		{

			for (itr = gpuResourceUtil.begin(); itr != gpuResourceUtil.end(); ++itr)
			{
				cerr<<"Time: "<<(*itr)->timestamp ;
				cerr<<" ,GPU_id: "<<(*itr)->gpuId <<"  ,Util_GPU: "<< (*itr)->utilGPU<<" (%)";
				cerr<<" ,PowerGPU: "<<(*itr)->power<<" (MW)";
				for (itr_cpu = cpuResourceUtil.begin(); itr_cpu != cpuResourceUtil.end(); ++itr_cpu)
				{
					if ( (*itr)->timestamp == (*itr_cpu)->timestamp ) 
					{
						cerr<<" ,CPU_id: "<<(*itr_cpu)->cpuId<< " ,Util_CPU: "<< (*itr_cpu)->utilCPU<< " (%)";
					}
				}

				for (itr_cpu_energy = cpuEnergy_vector.begin(); itr_cpu_energy != cpuEnergy_vector.end(); ++itr_cpu_energy)
				{
					if ( (*itr)->timestamp == (*itr_cpu_energy)->timestamp )
					{
						cerr<< " ,P0: " << (*itr_cpu_energy)->energyCPU[0]-energy[0] << " (UJ)";
						cerr<< " ,P1: " << (*itr_cpu_energy)->energyCPU[1]-energy[1] << " (UJ)";
						energy[0] = (*itr_cpu_energy)->energyCPU[0];
						energy[1] = (*itr_cpu_energy)->energyCPU[1];
					}
				}


				cerr<<endl<<endl;
			}
			gpuResourceUtil.clear();
			cpuResourceUtil.clear();
			cpuEnergy_vector.clear();


		}
		/*No GPUs, ONLY CPUs*/
		else
#endif
		{
			cout<<"NO GPU, only cpu"<<endl;
			cerr<<"Time: "<<cpuEnergy->timestamp;
			cerr<<" ,GPU_id: 1389 ,Util_GPU: 200 (%)";
			cerr<<" ,PowerGPU: 200 (MW)";
			for (itr_cpu = cpuResourceUtil.begin(); itr_cpu != cpuResourceUtil.end(); ++itr_cpu)
			{
				cerr<<" ,CPU_id: "<<(*itr_cpu)->cpuId<< " ,Util_CPU: "<< (*itr_cpu)->utilCPU<< " (%)";
				for (itr_cpu_energy = cpuEnergy_vector.begin(); itr_cpu_energy != cpuEnergy_vector.end(); ++itr_cpu_energy)
				{
					if ( (*itr_cpu)->timestamp == (*itr_cpu_energy)->timestamp && (*itr_cpu)->cpuId == 4 )
					{
						cerr<< " ,P0: " << (*itr_cpu_energy)->energyCPU[0]-energy[0] << " (UJ)";
						cerr<< " ,P1: " << (*itr_cpu_energy)->energyCPU[1]-energy[1] << " (UJ)";
						energy[0] = (*itr_cpu_energy)->energyCPU[0];
						energy[1] = (*itr_cpu_energy)->energyCPU[1];
					}
				}
			}
		}
	}
	//	result = nvmlShutdown();
	//	if (NVML_SUCCESS != result)
	//		printf("Failed to shutdown NVML: %s\n", nvmlErrorString(result));

	delete gpuInfo;
	delete cpuInfo;
	delete cpuEnergy;
	return 0;  
}
