#!/bin/bash
#########################################
	#	INPUTs 		#
	# 1. sampling time : 10000 is microsecond -> 0.01 secs
	# 2. monitoring cores : 4 (core1-core4)

#########################################

taskset -c 30 ./monitoringThread 10000 4 &> ../results/1.system_Info 
#cp 4.inputCPU_Util ../results/
#cp 1.inputGPU_Util ../results/
