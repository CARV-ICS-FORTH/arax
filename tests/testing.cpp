#define CATCH_CONFIG_MAIN
#include "testing.h"

int test_file_exists(const char *file)
{
    struct stat buf;

    return !stat(file, &buf);
}

int test_rename(const char *src, const char *dst)
{
    int ret = rename(src, dst);

    if (ret)
        printf("Renaming %s to %s failed:%s\n", src, dst, strerror(ret));

    return ret;
}

const char* test_create_config(size_t size)
{
    static char conf[2048];

    srand(system_process_id());

    uint64_t r = rand() * 68718952447ull;

    sprintf(conf, "shm_file test%016lx\nshm_size %lu\n", r, size);

    return conf;
}

const char *conf_backup = 0;

extern "C" {
const char* conf_get(const char *path);
void conf_set(const char *path, const char *conf_str);
}

/**
 * Backup current config, and clear config
 */
void test_backup_config()
{
    char *conf_path = utils_config_alloc_path(ARAX_CONFIG_FILE);

    conf_backup = conf_get(conf_path);
    conf_set(conf_path, "");
    utils_config_free_path(conf_path);
}

/**
 * Restore config to value saved by \c test_backup_config
 */
void test_restore_config()
{
    if (strlen(conf_backup)) {
        char *conf_path = utils_config_alloc_path(ARAX_CONFIG_FILE);
        conf_set(conf_path, conf_backup);
        utils_config_free_path(conf_path);
    }
}

/**
 * Open config file at ARAX_CONFIG_FILE.
 * \note use close() to close returned file descriptor.
 * @return File descriptor of the configuration file.
 */
int test_open_config()
{
    int fd;
    char *conf_file = utils_config_alloc_path(ARAX_CONFIG_FILE);

    fd = open(conf_file, O_RDWR | O_CREAT, 0666);
    REQUIRE(fd > 0);
    REQUIRE(system_file_size(conf_file) == lseek(fd, 0, SEEK_END));
    utils_config_free_path(conf_file);
    return fd;
}

pthread_t* spawn_thread(void *(func) (void *), void *data)
{
    pthread_t *thread = new pthread_t();

    REQUIRE(!!thread);
    pthread_create(thread, 0, func, data);
    return thread;
}

void wait_thread(pthread_t *thread)
{
    REQUIRE(!!thread);
    pthread_join(*thread, 0);
    delete thread;
}

int get_object_count(arax_object_repo_s *repo, arax_object_type_e type)
{
    int ret =
      arax_object_list_lock(repo, type)->length;

    arax_object_list_unlock(repo, type);
    return ret;
}

typedef struct
{
    int               tasks;
    arax_accel_type_e type;
    arax_accel_s *    accel;
    arax_pipe_s *     vpipe;
    pthread_t *       thread;
} n_task_handler_state;

#define SEC_IN_USEC (1000 * 1000)

void safe_usleep(int64_t us)
{
    struct timespec rqtp;

    if (us >= SEC_IN_USEC) // Time is >= 1s
        rqtp.tv_sec = us / SEC_IN_USEC;
    else
        rqtp.tv_sec = 0;

    us -= rqtp.tv_sec * SEC_IN_USEC; // Remove the 'full' seconds

    rqtp.tv_nsec = us * 1000;

    while (nanosleep(&rqtp, &rqtp) != 0);
}

void* balancer_thread(void *data)
{
    n_task_handler_state *state = (n_task_handler_state *) data;

    arax_pipe_s *vpipe = state->vpipe;

    while (state->tasks) {
        arax_vaccel_s *vac = arax_pipe_get_orphan_vaccel(vpipe);
        if (vac)
            arax_accel_set_physical(vac, state->accel);
    }

    return 0;
}

void* n_task_handler(void *data)
{
    n_task_handler_state *state = (n_task_handler_state *) data;

    arax_pipe_s *vpipe = arax_init();

    state->vpipe = vpipe;

    REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_VIRT_ACCEL) == 1);
    REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_PHYS_ACCEL) == 0);

    safe_usleep(100000);

    state->accel = arax_accel_init(vpipe, "Test",
        ANY, 1024 * 1024, 1024 * 1024);

    REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_PHYS_ACCEL) == 1);

    pthread_t *b_thread = spawn_thread(balancer_thread, data);

    while (state->tasks) {
        printf("%s(%d)\n", __func__, state->tasks);
        arax_accel_wait_for_task(state->accel);
        arax_vaccel_s **vacs;
        size_t nvacs = arax_accel_get_assigned_vaccels(state->accel, &vacs);

        for (int v = 0; v < nvacs; v++) {
            arax_vaccel_s *vac    = vacs[v];
            arax_task_msg_s *task = (arax_task_msg_s *) utils_queue_pop(arax_vaccel_queue(vac));
            if (task) {
                arax_proc_s *proc = (arax_proc_s *) task->proc;

                printf("Executing a '%s' task!\n", proc->obj.name);
                arax_task_mark_done(task, task_completed);
                state->tasks--;
            }
        }

        free(vacs);
    }
    printf("%s(%d)\n", __func__, state->tasks);

    arax_pipe_orphan_stop(vpipe);

    wait_thread(b_thread);

    arax_exit();
    return 0;
} // n_task_handler

void* handle_n_tasks(int tasks, arax_accel_type_e type)
{
    n_task_handler_state *state = new n_task_handler_state();

    state->tasks  = tasks;
    state->type   = type;
    state->thread = spawn_thread(n_task_handler, (void *) state);
    return state;
}

int handled_tasks(void *state)
{
    n_task_handler_state *handler_state = (n_task_handler_state *) state;

    wait_thread(handler_state->thread);

    int tasks = handler_state->tasks;

    arax_accel_release((arax_accel **) &(handler_state->accel));

    REQUIRE(get_object_count(&(handler_state->vpipe->objs), ARAX_TYPE_PHYS_ACCEL) == 0);
    REQUIRE(get_object_count(&(handler_state->vpipe->objs), ARAX_TYPE_TASK) == 0);

    delete handler_state;
    fprintf(stderr, "Tasks(%d)\n", tasks);
    return tasks == 0;
}

void test_common_setup(const char *conf)
{
    test_backup_config();
    unlink("/dev/shm/vt_test"); /* Start fresh */
    char *conf_path = utils_config_alloc_path(ARAX_CONFIG_FILE);

    conf_set(conf_path, conf);
    utils_config_free_path(conf_path);
}

void test_common_teardown()
{
    test_restore_config();
    // The shared segment should have been unlinked at this time.
    // If this crashes you are mostl likely missing a arax_exit().
    int err = unlink("/dev/shm/vt_test");

    if (err && errno != ENOENT) { // If unlink failed, and it wasnt because the file did not exist
        fprintf(stderr, "Shm file at '%s', not cleaned up but end of test(err:%d,%d)!\n", "/dev/shm/vt_test", err,
          errno);
    }
}

void arax_no_obj_leaks(arax_pipe_s *vpipe)
{
    // Check each type - this must be updated if new Object Types are added.
    REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_PHYS_ACCEL) == 0);
    REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_VIRT_ACCEL) == 0);
    REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_PROC) == 0);
    REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_DATA) == 0);
    REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_TASK) == 0);
    // Recheck with loop - this will always check all types
    for (arax_object_type_e type = ARAX_TYPE_PHYS_ACCEL; type < ARAX_TYPE_COUNT; type = (arax_object_type_e) (type + 1))
        REQUIRE(get_object_count(&(vpipe->objs), type) == 0);
}

arax_proc_s* create_proc(arax_pipe_s *vpipe, const char *name)
{
    arax_proc_s *proc;

    REQUIRE(!!vpipe);
    REQUIRE(!arax_proc_get(name) );
    proc = (arax_proc_s *) arax_proc_register(name);
    return proc;
}

static size_t initialy_available;

arax_pipe_s* arax_first_init()
{
    if (arax_clean())
        fprintf(stderr, "Warning, found and removed stale shm file!\n");

    arax_pipe_s *vpipe = arax_controller_init_start();

    REQUIRE(vpipe); // Get a pipe

    REQUIRE(vpipe->processes == 1); // Must be freshly opened

    REQUIRE(arax_pipe_have_orphan_vaccels(vpipe) == 0);

    arax_no_obj_leaks(vpipe);

    initialy_available = arax_pipe_get_available_size(vpipe);

    return vpipe;
}

void arax_final_exit(arax_pipe_s *vpipe)
{
    arax_no_obj_leaks(vpipe);

    REQUIRE(vpipe->processes == 1); // Only process

    if (arax_pipe_have_orphan_vaccels(vpipe)) {
        arax_vaccel_s *o = arax_pipe_get_orphan_vaccel(vpipe);
        fprintf(stderr, "Had orphan vaccel %p %s\n", o, o->obj.name);
    }

    REQUIRE(initialy_available == arax_pipe_get_available_size(vpipe));

    arax_exit();
}
