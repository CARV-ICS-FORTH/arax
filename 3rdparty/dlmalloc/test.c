#include <stdio.h>
#include <stdlib.h>
#include "malloc.h"
#define POOL_SIZE 1024*1024*1024
#define ALLOC_SIZE 1024
char * pool;
void ** pointers;	//Wont fill this, but better safe

int main(int argc,char *argv[])
{
	int allocs = 0;

	pool = malloc(POOL_SIZE);
	pointers = malloc((POOL_SIZE/ALLOC_SIZE)*sizeof(void*));

	mspace mem = create_mspace_with_base(pool,POOL_SIZE,1);
	while( (pointers[allocs]=mspace_malloc(mem,ALLOC_SIZE)) )
	{
		allocs++;
	}
	printf("Allocations: %d\n",allocs);
	mspace_bulk_free(mem,pointers,allocs);

	free(pointers);
	free(pool);

	return 0;
}
