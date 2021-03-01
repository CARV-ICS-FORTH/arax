#ifndef TESTING_HEADER
#define TESTING_HEADER
#include <check.h>
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
#include "vine_pipe.h"
#include "utils/system.h"
#include "utils/config.h"

int test_file_exists(char *file);

int test_rename(const char *src, const char *dst);

const char* test_create_config(size_t size);

/**
 * Backup current config VINE_CONFIG_FILE to ./vinetalk.bak.
 */
void test_backup_config();

/**
 * Restore ./vinetalk.bak. to VINE_CONFIG_FILE.
 */
void test_restore_config();

/**
 * Open config file at VINE_CONFIG_FILE.
 * \note use close() to close returned file descriptor.
 * @return File descriptor of the configuration file.
 */
int test_open_config();

pthread_t *spawn_thread(void *(func) (void *), void *data);

void wait_thread(pthread_t *thread);

int get_object_count(vine_object_repo_s *repo, vine_object_type_e type);

void safe_usleep(int64_t us);

void* n_task_handler(void *data);

void* handle_n_tasks(int tasks, vine_accel_type_e type);

int handled_tasks(void *state);

void test_common_setup();

void test_common_teardown();

void vine_no_obj_leaks(vine_pipe_s *vpipe);

vine_pipe_s* vine_first_init();

void vine_final_exit(vine_pipe_s *vpipe);

#endif /* ifndef TESTING_HEADER */
