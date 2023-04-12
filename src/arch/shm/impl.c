#include <arax.h>
#include <arax_pipe.h>
#include "core/arax_data.h"
#include "utils/config.h"
#include "utils/system.h"
#include "utils/timer.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

const char *arax_version = "VT_VERSION " ARAX_GIT_REV " - " ARAX_GIT_BRANCH;

struct
{
    arax_pipe_s *     vpipe;
    char              shm_file[1024];
    uint64_t          threads;
    uint64_t          instance_uid;
    uint64_t          task_uid;
    volatile uint64_t initialized;
    size_t            initialy_available;
    char *            config_path;
    int               fd;
} arax_state =
{ (void *) CONF_ARAX_MMAP_BASE, { '\0' }, 0, 0, 0, 0, 0, NULL, 0 };

#define arax_pipe_get() arax_state.vpipe

#define GO_FAIL(MSG)    ({ err = __LINE__; err_msg = MSG; goto FAIL; })

arax_pipe_s* _arax_init(int wait_controller)
{
    arax_pipe_s *shm_addr = 0;
    int err             = 0;
    size_t shm_size     = 0;
    size_t shm_off      = 0;
    int shm_trunc       = 0;
    int shm_ivshmem     = 0;
    int enforce_version = 0;
    const char *err_msg = "No Error Set";

    #ifdef MMAP_POPULATE
    mmap_flags |= MAP_POPULATE;
    #endif

    if (__sync_fetch_and_add(&(arax_state.threads), 1) != 0) { // I am not the first but stuff might not yet be initialized
        while (!arax_state.initialized);                       // wait for initialization
        goto GOOD;
    }

    arax_state.config_path = utils_config_alloc_path(ARAX_CONFIG_FILE);

    printf("Config:%s\n", ARAX_CONFIG_FILE);

    /* Required Confguration Keys */
    if (!utils_config_get_str(arax_state.config_path, "shm_file", arax_state.shm_file, 1024, 0) )
        GO_FAIL("No shm_file set in config");

    /* Default /4 of system memory*/
    shm_size = system_total_memory() / 4;

    utils_config_get_size(arax_state.config_path, "shm_size", &shm_size, shm_size);

    if (!shm_size || shm_size > system_total_memory() )
        GO_FAIL("shm_size exceeds system memory");

    /* Optional Confguration Keys */
    utils_config_get_size(arax_state.config_path, "shm_off", &shm_off, 0);
    utils_config_get_bool(arax_state.config_path, "shm_trunc", &shm_trunc, 1);
    utils_config_get_bool(arax_state.config_path, "shm_ivshmem", &shm_ivshmem, 0);
    utils_config_get_bool(arax_state.config_path, "enforce_version", &enforce_version, 1);

    if (shm_ivshmem) {
        shm_off  += 4096; /* Skip register section */
        shm_trunc = 0;    /* Don't truncate ivshm  device */
    }

    if (system_mmap((void **) &(arax_state.vpipe), &(arax_state.fd), arax_state.shm_file, shm_size, shm_off,
      shm_trunc))
    {
        if (arax_state.fd < 0)
            GO_FAIL("Could not open shm_file");
        if (arax_state.vpipe == NULL)
            GO_FAIL("Could not first mmap shm_file");
    }

    shm_addr = arax_pipe_mmap_address(arax_state.vpipe);

    if (arax_pipe_have_to_mmap(arax_state.vpipe, system_process_id())) {
        if (shm_addr != arax_state.vpipe) {
            munmap(arax_state.vpipe, arax_state.vpipe->shm_size); // unmap misplaced map.

            arax_state.vpipe = shm_addr; // Place mmap where we want it

            if (system_mmap((void **) &(arax_state.vpipe), &(arax_state.fd), arax_state.shm_file, shm_size, shm_off,
              0) || arax_state.vpipe != shm_addr)
                GO_FAIL("Could not mmap shm_file in proper address");
        }
    } else { // Proccess was already mmaped (although we forgot about it :-S
        arax_state.vpipe = shm_addr;
    }

    arax_state.vpipe = arax_pipe_init(arax_state.vpipe, shm_size, enforce_version);

    if (!arax_state.vpipe)
        GO_FAIL("Could not initialize arax_pipe");

    arax_state.initialy_available = arax_pipe_get_available_size(arax_state.vpipe);

    async_meta_init_always(&(arax_state.vpipe->async) );
    printf("ShmFile:%s\n", arax_state.shm_file);
    printf("ShmLocation:%p\n", arax_state.vpipe);
    printf("ShmSize:%zu\n", arax_state.vpipe->shm_size);
    arax_state.instance_uid = __sync_fetch_and_add(&(arax_state.vpipe->last_uid), 1);
    printf("InstanceUID:%zu\n", arax_state.instance_uid);
    arax_state.initialized = 1;
GOOD:
    if (wait_controller) {
        async_condition_lock(&(arax_state.vpipe->cntrl_ready_cond));
        while (arax_state.vpipe->cntrl_ready == 0)
            async_condition_wait(&(arax_state.vpipe->cntrl_ready_cond));
        async_condition_unlock(&(arax_state.vpipe->cntrl_ready_cond));
    }
    return arax_state.vpipe;

FAIL:
    printf("%c[31mprepare_arax Failed on line %d (conf:%s,file:%s,shm:%p)\n\
			Why:%s%c[0m\n", 27, err, ARAX_CONFIG_FILE, arax_state.shm_file,
      arax_state.vpipe, err_msg, 27);
    munmap(arax_state.vpipe, arax_state.vpipe->shm_size);
    exit(1);
} /* arax_task_init */

arax_pipe_s* arax_init()
{
    // All applications will wait for the controller/server initialization.
    return _arax_init(1);
}

arax_pipe_s* arax_controller_init_start()
{
    return _arax_init(0);
}

void arax_controller_init_done()
{
    arax_state.vpipe->cntrl_ready = 1;
    async_condition_notify(&(arax_state.vpipe->cntrl_ready_cond));
}

#undef GO_FAIL

uint64_t arax_instance_uid()
{
    return arax_state.instance_uid;
}

void arax_exit()
{
    int last;

    if (arax_state.vpipe) {
        if (__sync_fetch_and_add(&(arax_state.threads), -1) == 1) { // Last thread of process
            last = arax_pipe_exit(arax_state.vpipe);

            if (last) {
                size_t available = arax_pipe_get_available_size(arax_state.vpipe);
                arax_assert(available == arax_state.initialy_available);

                if (available != arax_state.initialy_available) {
                    printf("\033[1;31mERROR : shm LEAK !!\n\033[0m");
                }
            }

            arax_pipe_mark_unmap(arax_state.vpipe, system_process_id());
            munmap(arax_state.vpipe, arax_state.vpipe->shm_size);
            arax_state.vpipe = (void *) CONF_ARAX_MMAP_BASE;

            utils_config_free_path(arax_state.config_path);
            printf("arax_pipe_exit() = %d\n", last);
            close(arax_state.fd);
            arax_state.fd = 0;
            if (last) {
                if (!arax_clean() )
                    printf("Could not delete \"%s\"\n", arax_state.shm_file);
            }
        }
    } else {
        fprintf(stderr,
          "WARNING:arax_exit() called with no matching\
		call to arax_init()!\n");
    }
} /* arax_exit */

int arax_clean()
{
    char shm_file[1024];

    char *config_path = utils_config_alloc_path(ARAX_CONFIG_FILE);

    if (!utils_config_get_str(config_path, "shm_file", shm_file, 1024, 0) )
        arax_assert(!"No shm_file set in config");

    int ret = shm_unlink(shm_file);

    utils_config_free_path(config_path);

    return ret == 0;
}

void arax_accel_set_physical(arax_accel *vaccel, arax_accel *phys)
{
    arax_assert(phys);
    arax_assert(vaccel);
    arax_vaccel_s *acl = (arax_vaccel_s *) vaccel;

    arax_assert(acl);
    arax_accel_add_vaccel(phys, acl);
}

void arax_accel_list_free_pre_locked(arax_accel **accels);

int arax_accel_list(arax_accel_type_e type, int physical, arax_accel ***accels)
{
    arax_pipe_s *vpipe;
    utils_list_node_s *itr;
    utils_list_s *acc_list;

    arax_accel_s **acl = 0;
    int accel_count    = 0;
    arax_object_type_e ltype;


    if (physical)
        ltype = ARAX_TYPE_PHYS_ACCEL;
    else
        ltype = ARAX_TYPE_VIRT_ACCEL;

    vpipe = arax_pipe_get();

    acc_list =
      arax_object_list_lock(&(vpipe->objs), ltype);

    if (accels) { /* Want the accels */
        if (*accels)
            arax_accel_list_free_pre_locked(*accels);
        *accels = malloc( (acc_list->length + 1) * sizeof(arax_accel *) );
        acl     = (arax_accel_s **) *accels;
    }

    if (physical) {
        arax_accel_s *accel = 0;
        utils_list_for_each(*acc_list, itr){
            accel = (arax_accel_s *) itr->owner;
            if (!type || accel->type == type) {
                accel_count++;
                if (acl) {
                    arax_object_ref_inc(&(accel->obj));
                    *acl = accel;
                    acl++;
                }
            }
        }
    } else {
        arax_vaccel_s *accel = 0;
        utils_list_for_each(*acc_list, itr){
            accel = (arax_vaccel_s *) itr->owner;
            if (!type || accel->type == type) {
                accel_count++;
                if (acl) {
                    arax_object_ref_inc(&(accel->obj));
                    *acl = (arax_accel_s *) accel;
                    acl++;
                }
            }
        }
    }
    if (acl)
        *acl = 0;  // delimiter
    arax_object_list_unlock(&(vpipe->objs), ltype);

    return accel_count;
} /* arax_accel_list */

void arax_accel_list_free(arax_accel **accels)
{
    arax_object_s **itr = (arax_object_s **) accels;

    while (*itr) {
        arax_object_ref_dec(*itr);
        itr++;
    }
    free(accels);
}

void arax_accel_list_free_pre_locked(arax_accel **accels)
{
    arax_object_s **itr = (arax_object_s **) accels;

    while (*itr) {
        arax_object_ref_dec_pre_locked(*itr);
        itr++;
    }
    free(accels);
}

arax_accel_type_e arax_accel_type(arax_accel *accel)
{
    arax_accel_s *_accel;

    _accel = accel;

    return _accel->type;
}

arax_accel_state_e arax_accel_stat(arax_accel *accel, arax_accel_stats_s *stat)
{
    arax_accel_s *_accel;
    arax_accel_state_e ret;

    _accel = accel;

    switch (_accel->obj.type) {
        case ARAX_TYPE_PHYS_ACCEL:
            ret = arax_accel_get_stat(_accel, stat);
            break;
        case ARAX_TYPE_VIRT_ACCEL:
            ret = arax_vaccel_get_stat((arax_vaccel_s *) _accel, stat);
            break;
        default:
            ret = accel_failed; /* Not very 'correct' */
    }

    return ret;
}

int arax_accel_acquire_phys(arax_accel **accel)
{
    arax_pipe_s *vpipe;
    arax_accel_s *_accel;
    int return_value = 0;


    vpipe = arax_pipe_get();

    _accel = *accel;

    if (_accel->obj.type == ARAX_TYPE_PHYS_ACCEL) {
        *accel       = arax_vaccel_init(vpipe, "FILL", _accel->type, _accel);
        return_value = 1;
    }

    return return_value;
}

arax_accel* arax_accel_acquire_type(arax_accel_type_e type)
{
    arax_pipe_s *vpipe;
    arax_accel_s *_accel = 0;

    vpipe = arax_pipe_get();

    _accel = (arax_accel_s *) arax_vaccel_init(vpipe, "FILL", type, 0);

    return (arax_accel *) _accel;
}

void arax_accel_release(arax_accel **accel)
{
    arax_vaccel_s *_accel;

    _accel = *accel;

    switch (_accel->obj.type) {
        case ARAX_TYPE_PHYS_ACCEL:
        case ARAX_TYPE_VIRT_ACCEL:
            arax_object_ref_dec(&(_accel->obj));
            *accel = 0;
            return;

        default:
            arax_assert(!"Non accelerator type passed in arax_accel_release");
    }
}

arax_proc* arax_proc_register(const char *func_name)
{
    arax_pipe_s *vpipe;
    arax_proc_s *proc = 0;


    vpipe = arax_pipe_get();
    proc  = arax_pipe_find_proc(vpipe, func_name);

    if (!proc) { /* Proc has not been declared */
        proc = arax_proc_init(&(vpipe->objs), func_name);
    }

    return proc;
}

arax_proc* arax_proc_get(const char *func_name)
{
    arax_pipe_s *vpipe = arax_pipe_get();
    arax_proc_s *proc  = arax_pipe_find_proc(vpipe, func_name);

    if (proc)
        arax_object_ref_inc(&(proc->obj));
    else
        fprintf(stderr, "Proc %s not found!\n", func_name);

    return proc;
}

int arax_proc_put(arax_proc *func)
{
    arax_proc_s *proc = func;
    /* Decrease user count */
    int return_value = arax_object_ref_dec(&(proc->obj));

    return return_value;
}

int check_semantics(size_t in_count, arax_data **input, size_t out_count,
  arax_data **output)
{
    size_t io_cnt;
    size_t dup_cnt;
    size_t all_io = out_count + in_count;
    arax_data_s *temp_data_1 = 0;
    arax_data_s *temp_data_2 = 0;

    for (io_cnt = 0; io_cnt < all_io; io_cnt++) {
        // Choose from input or output
        if (io_cnt < in_count)
            temp_data_1 = input[io_cnt];
        else
            temp_data_1 = output[io_cnt - in_count];
        // check Validity temp_data_1
        if (!temp_data_1) {
            fprintf(stderr, "NULL input #%lu\n", io_cnt);
            return 0;
        }
        if (temp_data_1->obj.type != ARAX_TYPE_DATA) {
            fprintf(stderr, "Input #%lu not valid data\n", io_cnt);
            return 0;
        }
        // Check duplicates
        for (dup_cnt = 0; dup_cnt < all_io; dup_cnt++) {
            // Choose from input or output
            if (dup_cnt < in_count)
                temp_data_2 = input[dup_cnt];
            else
                temp_data_2 = output[dup_cnt - in_count];
            // check Validity temp_data_2
            if (!temp_data_2) {
                fprintf(stderr, "NULL input #%lu\n", dup_cnt);
                return 0;
            }
            if (temp_data_2->obj.type != ARAX_TYPE_DATA) {
                fprintf(stderr, "Input #%lu not valid data\n", dup_cnt);
                return 0;
            }
        }
    }
    return 1;
} /* check_semantics */

arax_task* arax_task_issue(arax_accel *accel, arax_proc *proc, const void *host_init, size_t host_size,
  size_t in_count, arax_data **dev_in, size_t out_count,
  arax_data **dev_out)
{
    // printf("%s %s\n",__func__, ((arax_proc_s*)proc)->obj.name) ;

    arax_pipe_s *vpipe = arax_pipe_get();
    arax_task_msg_s *task;

    arax_assert(check_semantics(in_count, dev_in, out_count, dev_out));

    task = arax_task_alloc(vpipe, accel, proc, host_size, in_count, dev_in, out_count, dev_out);

    arax_assert(task);

    if (host_size && host_init)
        memcpy(arax_task_host_data(task, host_size), host_init, host_size);

    arax_task_submit(task);

    return task;
} /* arax_task_issue */

arax_task_state_e arax_task_issue_sync(arax_accel *accel, arax_proc *proc, void *host_init,
  size_t host_size, size_t in_count, arax_data **dev_in, size_t out_count,
  arax_data **dev_out)
{
    arax_task *task = arax_task_issue(accel, proc, host_init, host_size, in_count, dev_in, out_count, dev_out);
    arax_task_state_e status = arax_task_wait(task);

    arax_task_free(task);

    return status;
}

arax_task_state_e arax_task_stat(arax_task *task, arax_task_stats_s *stats)
{
    arax_task_msg_s *_task = task;
    arax_task_state_e ret  = 0;

    ret = _task->state;

    if (stats)
        memcpy(stats, &(_task->stats), sizeof(*stats));

    return ret;
}

arax_task_state_e arax_task_wait(arax_task *task)
{
    arax_task_msg_s *_task = task;

    arax_task_wait_done(_task);

    return _task->state;
}

void arax_task_free(arax_task *task)
{
    arax_task_msg_s *_task = task;

    arax_object_ref_dec(&(_task->obj));
}

arax_buffer_s ARAX_BUFFER(size_t size)
{
    arax_pipe_s *vpipe = arax_pipe_get();

    arax_data_s *vd = arax_data_init(vpipe, size);

    return vd;
}
