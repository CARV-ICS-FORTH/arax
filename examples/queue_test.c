#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/mman.h>

#include "atring.h"

int thread_ops;

void usage(char *name)
{
	printf("Usage:\n" "  %s ROLE THREADS THREAD_OPS SHM_FILE SHM_SIZE\n"
	       "    ROLE       : 0 for producer, 1 for consumer.\n"
	       "    THREADS : Number of threads to use.\n"
	       "    THREADS_OPS: Number of operations(push,pop) to be done by every thread.\n"
	       "    SHM_FILE   : Shared memory file(size > 0).\n"
	       "    SHM_SIZE   : Shared memory size(bytes).\n",
	        name);
}

pthread_t* spawn_threads(int count, void* (*func)(void*), void *data)
{
	pthread_t *threadp;
	int       cnt;

	threadp = malloc(sizeof(pthread_t)*count);
	for (cnt = 0; cnt < count; cnt++)
		pthread_create(threadp+cnt, 0, func, data);
	return threadp;
}

void* producer(void *data)
{
	long int my_ops = thread_ops;
	atring_s *ring  = data;

	while (my_ops) {
		while ( !atring_push( ring, (void*)(my_ops) ) )
			;
		my_ops--;
	}
	return 0;
}

void* consumer(void *data)
{
	long int my_ops = thread_ops;
	atring_s *ring  = data;

	while (my_ops) {
		do {
			data = atring_pop(ring);
		} while (!data);
		/* printf("Pop result = %d\n", (int)data); */
		my_ops--;
	}
	return 0;
}

atring_s* init_ring(char *file, int size, int role)
{
	atring_s *ring = 0;
	int      fd    = shm_open(file, O_CREAT|O_RDWR, S_IRWXU);

	if (fd < 0)
		return 0;

	if ( ftruncate(fd, size) )
		return 0;

	ring = mmap(0, size, PROT_READ|PROT_WRITE|PROT_EXEC, MAP_SHARED, fd, 0);

	if (!ring)
		return 0;

	if (!role)
		ring = atring_init(ring, size);

	return ring;
}

void wait_threads(pthread_t *threadp, int threads)
{
	while (threads--)
		pthread_join(threadp[threads], 0);
}

int main(int argc, char *argv[])
{
	if (argc != 6) {
		printf("Not enough arguments(%d!=6).\n", argc);
FAIL: usage(argv[0]);
		return -1;
	}

	int role    = atoi(argv[1]);
	int threads = atoi(argv[2]);

	thread_ops = atoi(argv[3]);

	int       shm_size = atoi(argv[5]);
	pthread_t *threadp;
	atring_s  *ring = 0;


	if ( !(thread_ops >= threads && threads > 0) ) {
		printf("Improper setting of THREADS or THREAD_OPS.\n");
		goto FAIL;
	}

	ring = init_ring(argv[4], shm_size, role);

	if (!ring) {
		printf("Ring allocation failed.\n");
		goto FAIL;
	}

	printf( "New atring at %p(%d slots)\n", ring, atring_free_slots(ring) );

	printf("Starting %d %s threads each doing %d operations.\n", threads,
	       (role) ? "consumer" : "producer", thread_ops);
	threadp = spawn_threads(threads, (role) ? consumer : producer, ring);
	wait_threads(threadp, threads);
	printf("All threads completed.\n");
	return 0;
}
