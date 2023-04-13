#ifndef TESTING_HEADER
#define TESTING_HEADER
#include <catch2/catch.hpp>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/stat.h>
#include <conf.h>
#include "arax_pipe.h"
#include "utils/system.h"
#include "utils/config.h"

int test_file_exists(const char *file);

int test_rename(const char *src, const char *dst);

const char* test_create_config(size_t size);

/**
 * Backup current config ARAX_CONFIG_FILE to ./arax.bak.
 */
void test_backup_config();

/**
 * Restore ./arax.bak. to ARAX_CONFIG_FILE.
 */
void test_restore_config();

/**
 * Open config file at ARAX_CONFIG_FILE.
 * \note use close() to close returned file descriptor.
 * @return File descriptor of the configuration file.
 */
int test_open_config();

pthread_t *spawn_thread(void *(func) (void *), void *data);

void wait_thread(pthread_t *thread);

int get_object_count(arax_object_repo_s *repo, arax_object_type_e type);

void safe_usleep(int64_t us);

void* n_task_handler(void *data);

void* handle_n_tasks(int tasks, arax_accel_type_e type);

int handled_tasks(void *state);

void test_common_setup(const char *conf);

void test_common_teardown();

void arax_no_obj_leaks(arax_pipe_s *vpipe);

arax_proc_s* create_proc(arax_pipe_s *vpipe, const char *name);

arax_pipe_s* arax_first_init();

void arax_final_exit(arax_pipe_s *vpipe);

#endif /* ifndef TESTING_HEADER */
