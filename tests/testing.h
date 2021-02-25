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

static int __attribute__( (unused) ) test_file_exists(char *file)
{
    struct stat buf;

    return !stat(file, &buf);
}

static int test_rename(const char *src, const char *dst)
{
    int ret = rename(src, dst);

    if (ret)
        printf("Renaming %s to %s failed:%s\n", src, dst, strerror(ret));

    return ret;
}

/**
 * Backup current config VINE_CONFIG_FILE to ./vinetalk.bak.
 */
static void __attribute__( (unused) ) test_backup_config()
{
    char *conf_file = utils_config_alloc_path(VINE_CONFIG_FILE);

    if (test_file_exists(conf_file) ) {
        fprintf(stderr, "Backup config file '%s' -> vinetalk.bak\n", conf_file);
        ck_assert(!test_rename(conf_file, "vinetalk.bak") ); /* Keep old
                                                              * file */
    }

    ck_assert_int_eq(system_file_size(conf_file), 0);

    utils_config_free_path(conf_file);
}

/**
 * Restore ./vinetalk.bak. to VINE_CONFIG_FILE.
 */
static void __attribute__( (unused) ) test_restore_config()
{
    char *conf_file = utils_config_alloc_path(VINE_CONFIG_FILE);

    if (test_file_exists("vinetalk.bak") ) {
        fprintf(stderr, "Restore config file vinetalk.bak -> '%s'\n", conf_file);
        ck_assert(!test_rename("vinetalk.bak", conf_file) );
    } else {
        if (test_file_exists(conf_file) )
            ck_assert(!unlink(conf_file) );  /* Remove test file*/
    }
    utils_config_free_path(conf_file);
}

/**
 * Open config file at VINE_CONFIG_FILE.
 * \note use close() to close returned file descriptor.
 * @return File descriptor of the configuration file.
 */
static int __attribute__( (unused) ) test_open_config()
{
    int fd;
    char *conf_file = utils_config_alloc_path(VINE_CONFIG_FILE);

    fd = open(conf_file, O_RDWR | O_CREAT, 0666);
    ck_assert_int_gt(fd, 0);
    ck_assert_int_eq(system_file_size(conf_file), lseek(fd, 0, SEEK_END));
    utils_config_free_path(conf_file);
    return fd;
}

static __attribute__( (unused) ) pthread_t* spawn_thread(void *(func) (void *), void *data)
{
    pthread_t *thread = malloc(sizeof(*thread));

    ck_assert(!!thread);
    pthread_create(thread, 0, func, data);
    return thread;
}

static __attribute__( (unused) ) void wait_thread(pthread_t *thread)
{
    ck_assert(!!thread);
    pthread_join(*thread, 0);
    free(thread);
}

static __attribute__( (unused) ) int get_object_count(vine_object_repo_s *repo, vine_object_type_e type)
{
    int ret =
      vine_object_list_lock(repo, type)->length;

    vine_object_list_unlock(repo, type);
    return ret;
}

typedef struct
{
    int               tasks;
    vine_accel_type_e type;
    vine_accel_s *    accel;
    vine_pipe_s *     vpipe;
} n_task_handler_state;

#define SEC_IN_USEC (1000 * 1000)

static __attribute__( (unused) ) void safe_usleep(int64_t us)
{
    struct timespec rqtp;

    if (us >= SEC_IN_USEC) // Time is >= 1s
        rqtp.tv_sec = us / SEC_IN_USEC;
    else
        rqtp.tv_sec = 0;

    us -= rqtp.tv_sec * SEC_IN_USEC; // Remote the 'full' seconds

    rqtp.tv_nsec = us * 1000;

    while (nanosleep(&rqtp, &rqtp) != 0);
}

void* n_task_handler(void *data)
{
    n_task_handler_state *state = data;

    vine_pipe_s *vpipe = vine_talk_init();

    state->vpipe = vpipe;

    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_VIRT_ACCEL), 1);
    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL), 0);

    safe_usleep(100000);

    state->accel = vine_accel_init(vpipe, "Test",
        ANY, 1024 * 1024, 1024 * 1024);

    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL), 1);

    while (state->tasks) {
        printf("%s(%d)\n", __func__, state->tasks);
        vine_pipe_wait_for_task_type_or_any_assignee(vpipe, state->type, 0);
        utils_list_s *vacs = vine_object_list_lock(&(vpipe->objs), VINE_TYPE_VIRT_ACCEL);
        utils_list_node_s *vac_node;
        vine_vaccel_s *vac;
        vine_task_msg_s *task;
        utils_list_for_each(*vacs, vac_node)
        {
            vac = vac_node->owner;
            vine_accel_set_physical(vac, state->accel);
            ck_assert_int_eq(vac->obj.type, VINE_TYPE_VIRT_ACCEL);
            task = utils_queue_pop(vine_vaccel_queue(vac));
            ck_assert(task);
            vine_proc_s *proc = task->proc;

            printf("Executing a '%s' task!\n", proc->obj.name);
            vine_task_mark_done(task, task_completed);
        }
        vine_object_list_unlock(&(vpipe->objs), VINE_TYPE_VIRT_ACCEL);
        state->tasks--;
    }
    printf("%s(%d)\n", __func__, state->tasks);

    return 0;
} // n_task_handler

static __attribute__( (unused) ) void* handle_n_tasks(int tasks, vine_accel_type_e type)
{
    n_task_handler_state *state = malloc(sizeof(*state));

    state->tasks = tasks;
    state->type  = type;
    spawn_thread(n_task_handler, (void *) state);
    return state;
}

static int __attribute__( (unused) ) handled_tasks(void *state)
{
    n_task_handler_state *handler_state = (n_task_handler_state *) state;
    int tasks = handler_state->tasks;

    vine_accel_release((vine_accel **) &(handler_state->accel));

    ck_assert_int_eq(get_object_count(&(handler_state->vpipe->objs), VINE_TYPE_PHYS_ACCEL), 0);

    free(handler_state);
    fprintf(stderr, "Tasks(%d)\n", tasks);
    return tasks == 0;
}

static void __attribute__( (unused) ) test_common_setup()
{
    test_backup_config();
    unlink("/dev/shm/vt_test"); /* Start fresh */
}

static void __attribute__( (unused) ) test_common_teardown()
{
    test_restore_config();
    // The shared segment should have been unlinked at this time.
    // If this crashes you are mostl likely missing a vine_talk_exit().
    int err = unlink("/dev/shm/vt_test");

    if (err && errno != ENOENT) // If unlink failed, and it wasnt because the file did not exist
        fprintf(stderr, "Shm file at '%s', not cleaned up but end of test(err:%d,%d)!\n", "/dev/shm/vt_test", err,
          errno);
}

static __attribute__( (unused) ) void vine_no_obj_leaks(vine_pipe_s *vpipe)
{
    // Check each type - this must be updated if new Object Types are added.
    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL), 0);
    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_VIRT_ACCEL), 0);
    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PROC), 0);
    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_DATA), 0);
    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_TASK), 0);
    // Recheck with loop - this will always check all types
    for (vine_object_type_e type = VINE_TYPE_PHYS_ACCEL; type < VINE_TYPE_COUNT; type++)
        ck_assert_int_eq(get_object_count(&(vpipe->objs), type), 0);
}

static __attribute__( (unused) ) vine_pipe_s* vine_first_init()
{
    if (vine_talk_clean())
        fprintf(stderr, "Warning, found and removed stale shm file!\n");

    vine_pipe_s *vpipe = vine_talk_init();

    ck_assert(vpipe); // Get a pipe

    ck_assert_int_eq(vpipe->processes, 1); // Must be freshly opened

    vine_no_obj_leaks(vpipe);

    return vpipe;
}

static __attribute__( (unused) ) void vine_final_exit(vine_pipe_s *vpipe)
{
    vine_no_obj_leaks(vpipe);

    ck_assert_int_eq(vpipe->processes, 1); // Only user

    vine_talk_exit();
}

#endif /* ifndef TESTING_HEADER */
