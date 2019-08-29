#include "vine_assert.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <execinfo.h>
#include "system.h"

#define VINE_FILE_PREFIX_LEN (strlen(__FILE__)-23)

void _vine_assert(int value,const char * expr,const char * file,int line)
{
	if(!value)
		return;
	void * bt[128];
	char ** bt_syms;
	int bt_size = 128;
	int bt_indx;

	bt_size = backtrace(bt, bt_size);
	bt_syms = backtrace_symbols(bt, bt_size);

	for(bt_indx = 1 ; bt_indx < bt_size ; bt_indx++)
	{
		fprintf(stderr,"%s\n",bt_syms[bt_indx]);
	}
	fprintf(stderr,"vine_assert(%s) @ %s:%d\n",expr,file+VINE_FILE_PREFIX_LEN,line);
	abort();
}
