#!/bin/bash

# Start arax_controller
# If ARAX_CONFIG eviroment var is defined use to initialize arax configuration
# If ARAXCNTRL_CONFIG enviroment var is defined use to initialize arax controller configuration

if [ -z "${ARAX_CONFIG}" ]
then
	echo "No ARAX_CONFIG provided"
else
	echo ${ARAX_CONFIG} > ~/.arax
fi

if [ -z "${ARAXCNTRL_CONFIG}" ]
then
	echo "No ARAXCNTRL_CONFIG provided"
else
	echo ${ARAXCNTRL_CONFIG} > /ci.conf
fi

LD_LIBRARY_PATH="" arax_controller /ci.conf &
echo $! > /var/run/cntrl
sleep 1
