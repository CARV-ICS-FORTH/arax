#include "btgen.h"
#include "config.h"
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <execinfo.h>

#include "conf.h"

struct SigRemap
{
	struct  sigaction prev;
	struct  sigaction vine;
	int signum;
};

#define CAPRTURED_SIGNAL_COUNT 5

int captured_signals[CAPRTURED_SIGNAL_COUNT] = 
{SIGILL,SIGTRAP,SIGABRT,SIGBUS,SIGSEGV};

struct SigRemap sig_remap[CAPRTURED_SIGNAL_COUNT];

int bt_fd = -1;
char bt_fname[64];

void bt_handler(int signum);

void utils_bt_init()
{
	int cnt = 0;

	for(cnt = 0 ; cnt < CAPRTURED_SIGNAL_COUNT ; cnt++)
	{
		sig_remap[cnt].signum = captured_signals[cnt];
		sig_remap[cnt].vine.sa_handler = bt_handler;

		if(sigaction(SIGBUS,&(sig_remap[cnt].vine),&(sig_remap[cnt].prev)) < 0)
		{
			fprintf(stderr,"Failed to register bt_handler for SIGBUS.");
			return;
		}
	}

	while(bt_fd < 0)
	{

		sprintf(bt_fname,"/tmp/vine_bt%d.lzr",cnt++);
		bt_fd = open(bt_fname,O_CREAT|O_APPEND|O_RDWR|O_EXCL,S_IRWXU);
	}
	write(bt_fd,"Git Revision: ",14);
	write(bt_fd,VINE_TALK_GIT_REV,strlen(VINE_TALK_GIT_REV));
	write(bt_fd,"\nBacktrace:\n",12);
}

void bt_handler(int signum)
{
	void * bt[32];
	int bt_size = 32;
	bt_size = backtrace(bt, bt_size);
	backtrace_symbols_fd(bt,bt_size,bt_fd);
	for(bt_size = 0 ; bt_size < CAPRTURED_SIGNAL_COUNT ; bt_size++)
	{
		if(sig_remap[bt_size].signum == signum)
		{
			sig_remap[bt_size].prev.sa_handler(signum);
			break;
		}
	}
}

void utils_bt_exit()
{
	close(bt_fd);
	remove(bt_fname);
}
