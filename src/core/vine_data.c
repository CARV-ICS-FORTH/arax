#include "vine_data.h"
#include "vine_pipe.h"
#include "vine_ptr.h"
#include <string.h>
#include <stdlib.h>
#include "utils/system.h"

// #define printd(...) fprintf(__VA_ARGS__)
#define  printd(...)

#define VDFLAG(DATA, FLAG)  (DATA->flags & FLAG)// ((size_t*)BUFF-1) same pointer arithmetics//
#define VD_BUFF_OWNER(BUFF) *(vine_data_s **) ((char *) BUFF - sizeof(size_t *))

vine_data_s* vine_data_init(vine_pipe_s *vpipe, void *user, size_t size)
{
    return vine_data_init_aligned(vpipe, user, size, 1);
} /* vine_data_init */

vine_data_s* vine_data_init_aligned(vine_pipe_s *vpipe, void *user, size_t size, size_t align)
{
    vine_data_s *data;
    size_t alloc_size = sizeof(vine_data_s) + VINE_BUFF_ALLOC_SIZE(size, align);

    vine_assert(align);

    data = (vine_data_s *) vine_object_register(&(vpipe->objs),
        VINE_TYPE_DATA,
        "UNUSED", alloc_size, 1);

    if (!data)     // GCOV_EXCL_LINE
        return 0;  // GCOV_EXCL_LINE

    uint64_t *buff_ptr = (uint64_t *) (data + 1); // Skip the struct

    buff_ptr++; // Skip a back_pointer
    buff_ptr = (uint64_t *) (((char *) buff_ptr) + align - (((size_t) buff_ptr) % align));// Align ptr;
    vine_data_s **back_pointer = (vine_data_s **) (buff_ptr - 1);

    *back_pointer = data;

    data->user   = user;
    data->size   = size;
    data->buffer = buff_ptr;
    data->align  = align;
    data->flags  = 0;
    async_completion_init(&(data->obj.repo->pipe->async), &(data->ready));

    VINE_THROTTLE_DEBUG_PRINT("%s(%p,Size:%lu,Align:%lu,Buffer:%lu,Total:%lu) ^^^^^\n", __func__, data, data->size,
      data->align, VINE_DATA_ALLOC_SIZE(data), VINE_DATA_ALLOC_SIZE(data) + sizeof(*data));

    return data;
} /* vine_data_init_aligned */

void vine_data_check_flags(vine_data_s *data)
{
    switch (data->flags) {
        case NONE_SYNC:
        case USER_SYNC:
        case SHM_SYNC:
        case USER_SYNC | SHM_SYNC:
        case REMT_SYNC:
        case REMT_SYNC | SHM_SYNC:
        case ALL_SYNC:
            return;

        default: // GCOV_EXCL_START
            fprintf(stderr, "%s(%p): Inconsistent data flags %lu\n", __func__, data, data->flags);
            vine_assert(!"Inconsistent data flags");
            // GCOV_EXCL_STOP
    }
}

void vine_data_memcpy(vine_accel *accel, vine_data_s *dst, vine_data_s *src, int block)
{
    vine_assert_obj(dst, VINE_TYPE_DATA);
    vine_assert_obj(src, VINE_TYPE_DATA);

    if (dst == src)
        return;

    if (vine_data_size(dst) != vine_data_size(src)) {
        fprintf(stderr, "%s(%p,%p): Size mismatch (%lu,%lu)\n", __func__, dst, src, vine_data_size(dst),
          vine_data_size(src));
        vine_assert(!"Size mismatch");
    }
    fprintf(stderr, "%s(%p,%p)[%lu,%lu]\n", __func__, dst, src, dst->flags, src->flags);

    vine_data_sync_from_remote(accel, src, block);

    memcpy(vine_data_deref(dst), vine_data_deref(src), vine_data_size(src));

    vine_data_modified(dst, SHM_SYNC);

    vine_data_sync_to_remote(accel, dst, block);
}

#define TYPE_MASK(A, B) ( ( (A) *VINE_TYPE_COUNT ) + (B) )

void vine_data_migrate_accel(vine_data_s *data, vine_accel *accel)
{
    vine_assert_obj(data, VINE_TYPE_DATA);
    vine_assert(accel); // Must be given a valid accelerator

    if (data->accel == accel) // Already assigned to accel - nop
        return;

    if (!data->accel) {             // Unassinged data - assign and increase accel
        vine_assert(!data->remote); // How can you have a remote ptr, with no accel?
        vine_object_ref_inc(((vine_object_s *) accel));
        data->accel = accel;
        return;
    }

    vine_object_type_e data_accel_type  = ((vine_object_s *) (data->accel))->type;
    vine_object_type_e accel_accel_type = ((vine_object_s *) (accel))->type;

    switch (TYPE_MASK(data_accel_type, accel_accel_type) ) {
        case TYPE_MASK(VINE_TYPE_VIRT_ACCEL, VINE_TYPE_VIRT_ACCEL): {
            vine_vaccel_s *dvac = (vine_vaccel_s *) (data->accel);
            vine_vaccel_s *avac = (vine_vaccel_s *) (accel);
            if (dvac->phys == avac->phys) { // Both use the same physical accel - just migrate ref counts
                vine_object_ref_dec(((vine_object_s *) dvac));
                vine_object_ref_inc(((vine_object_s *) avac));
                data->accel = accel;
                return;
            } else { // Device 2 Device migration - migrate data and ref counts
                fprintf(stderr, "Different device data migration!\n");
                vine_assert(!"vine_data_migrate_accel: D2D not implemented!");
            }
        }
        break;
        default: {
            fprintf(stderr, "%s():Data migration not implemented(%s:%s,%s:%s)!\n",
              __func__,
              vine_accel_type_to_str(data_accel_type), ((vine_object_s *) (data->accel))->name,
              vine_accel_type_to_str(accel_accel_type), ((vine_object_s *) (accel))->name
            );
            vine_assert(!"No migration possible");
        }
        break;
    }
} /* vine_data_migrate_accel */

void vine_data_allocate_remote(vine_data_s *data, vine_accel *accel)
{
    vine_assert_obj(data, VINE_TYPE_DATA);
    vine_assert_obj(accel, VINE_TYPE_VIRT_ACCEL);

    if ( ((vine_vaccel_s *) accel)->type == CPU)
        return;  // CPU does not have a 'remote', so nothing to do

    if (data->remote) {
        vine_assert(data->accel); // Allocate remote must have accel
        return;                   // Nothing left to do
    }

    VINE_THROTTLE_DEBUG_PRINT("%s(%p) - start\n", __func__, data);

    void *args[1] = { data };
    vine_proc_s *alloc_data = vine_proc_get("alloc_data");
    vine_task_msg_s *task   = vine_task_issue(accel, alloc_data, args, sizeof(args), 0, 0, 0, 0);

    vine_assert(vine_task_wait(task) == task_completed);
    vine_task_free(task);
    vine_assert(data->remote); // Ensure remote was allocated

    if (data->accel != accel) {
        data->accel = accel;
        vine_object_ref_inc(accel);
    }

    vine_accel_size_dec(((vine_vaccel_s *) accel)->phys, vine_data_size(data));
    VINE_THROTTLE_DEBUG_PRINT("%s(%p) - end\n", __func__, data);
}

void vine_data_arg_init(vine_data_s *data, vine_accel *accel)
{
    // check errors
    vine_assert_obj(data, VINE_TYPE_DATA);
    vine_assert(accel);

    vine_data_migrate_accel(data, accel);

    async_completion_init(&(data->obj.repo->pipe->async), &(data->ready));
}

void vine_data_input_init(vine_data_s *data, vine_accel *accel)
{
    // check errors
    vine_assert_obj(data, VINE_TYPE_DATA);
    vine_assert(accel);

    vine_object_ref_inc(&(data->obj));

    vine_data_migrate_accel(data, accel);

    async_completion_init(&(data->obj.repo->pipe->async), &(data->ready));
}

void vine_data_output_init(vine_data_s *data, vine_accel *accel)
{
    // check errors
    vine_assert_obj(data, VINE_TYPE_DATA);
    vine_assert(accel);

    vine_object_ref_inc(&(data->obj));

    vine_data_migrate_accel(data, accel);

    async_completion_init(&(data->obj.repo->pipe->async), &(data->ready));
}

void vine_data_output_done(vine_data_s *data)
{
    vine_assert_obj(data, VINE_TYPE_DATA);
    // Invalidate on all levels except accelerator memory.
    vine_data_modified(data, REMT_SYNC);
}

size_t vine_data_size(vine_data *data)
{
    vine_assert_obj(data, VINE_TYPE_DATA);
    vine_data_s *vdata;

    vdata = data;
    return vdata->size;
}

void* vine_data_deref(vine_data *data)
{
    vine_assert_obj(data, VINE_TYPE_DATA);
    vine_data_s *vdata;

    vdata = (vine_data_s *) data;

    VINE_THROTTLE_DEBUG_PRINT("%s(%p) - start\n", __func__, data);

    VINE_THROTTLE_DEBUG_PRINT("%s(%p) - end\n", __func__, data);

    return vdata->buffer;
}

vine_data* vine_data_ref(void *data)
{
    vine_assert(data);

    vine_data_s *vdata = VD_BUFF_OWNER(data);

    // GCOV_EXCL_START
    if (!vdata)
        return 0;

    if (!vine_ptr_valid(vdata))
        return 0;

    if (vdata->obj.type != VINE_TYPE_DATA)
        return 0;

    // GCOV_EXCL_STOP

    return vdata;
}

vine_data* vine_data_ref_offset(vine_pipe_s *vpipe, void *data)
{
    vine_assert(data);
    vine_assert(vine_ptr_valid(data));
    // Might be at the start
    vine_data_s *ret = vine_data_ref(data);

    if (ret)
        return ret;

    utils_list_s *datas = vine_object_list_lock(&(vpipe->objs), VINE_TYPE_DATA);
    utils_list_node_s *itr;

    utils_list_for_each(*datas, itr)
    {
        vine_data_s *vd = (vine_data_s *) (itr->owner);
        void *start     = vine_data_deref(vd);
        void *end       = start + vine_data_size(vd);

        if (data > start && data < end) {
            ret = vd;
            break; // Found it!
        }
    }

    vine_object_list_unlock(&(vpipe->objs), VINE_TYPE_DATA);

    return ret;
}

void vine_data_mark_ready(vine_pipe_s *vpipe, vine_data *data)
{
    vine_assert_obj(data, VINE_TYPE_DATA);
    vine_data_s *vdata;

    vdata = (vine_data_s *) data;
    async_completion_complete(&(vdata->ready));
}

int vine_data_check_ready(vine_pipe_s *vpipe, vine_data *data)
{
    vine_assert_obj(data, VINE_TYPE_DATA);
    vine_data_s *vdata;
    int return_val;

    vdata      = offset_to_pointer(vine_data_s *, vpipe, data);
    return_val = async_completion_check(&(vdata->ready));

    return return_val;
}

void vine_data_free(vine_data *data)
{
    vine_assert_obj(data, VINE_TYPE_DATA);
    vine_data_s *vdata;

    vdata = (vine_data_s *) data;
    vine_object_ref_dec(&(vdata->obj));
}

void vine_data_shm_sync(vine_accel *accel, const char *func, vine_data_s *data, int block)
{
    vine_assert_obj(data, VINE_TYPE_DATA);
    if (data->remote == vine_data_deref(data)) // Remote points to shm buffer
        return;

    void *args[2] = { data, (void *) (size_t) block };

    vine_accel_type_e type = ((vine_vaccel_s *) accel)->type;
    vine_proc_s *proc      = vine_proc_get(func);

    if (!vine_proc_get_functor(proc, type))
        return;

    vine_task_msg_s *task = vine_task_issue(accel, proc, args, sizeof(void *) * 2, 0, 0, 0, 0);

    if (block) {
        vine_assert(vine_task_wait(task) == task_completed);
        vine_task_free(task);
    }
}

/*
 * Send user data to the remote
 */
void vine_data_sync_to_remote(vine_accel *accel, vine_data *data, int block)
{
    vine_assert_obj(data, VINE_TYPE_DATA);
    vine_data_s *vdata;

    vdata = (vine_data_s *) data;

    vine_data_check_flags(data); // Ensure flags are consistent

    vine_data_migrate_accel(vdata, accel);

    switch (vdata->flags) {
        case USER_SYNC: // usr->shm
            vine_assert(vdata->user && "Attempting to vine_data_sync_to_remote from NULL user ptr");
            memcpy(vine_data_deref(vdata), vdata->user, vdata->size);
            vdata->flags |= SHM_SYNC;
        case USER_SYNC | SHM_SYNC:
        case SHM_SYNC: // shm->rmt
        case REMT_SYNC | SHM_SYNC:
        case REMT_SYNC:
        case ALL_SYNC:
            break; // All set
        default:   // GCOV_EXCL_START
            fprintf(stderr, "%s(%p) unexpected flags %lu!\n", __func__, data, vdata->flags);
            vine_assert(!"Uxpected flags");
            break;
        case NONE_SYNC:
            fprintf(stderr, "%s(%p) called with uninitialized buffer!\n", __func__, data);
            vine_assert(!"Uninitialized buffer");
            // GCOV_EXCL_STOP
    }

    vine_data_check_flags(data); // Ensure flags are consistent
} /* vine_data_sync_to_remote */

/*
 * Get remote data to user
 */
void vine_data_sync_from_remote(vine_accel *accel, vine_data *data, int block)
{
    vine_assert_obj(data, VINE_TYPE_DATA);
    vine_data_s *vdata;

    vdata = (vine_data_s *) data;

    vine_data_check_flags(data); // Ensure flags are consistent

    vine_data_migrate_accel(vdata, accel);

    switch (vdata->flags) {
        case REMT_SYNC: // rmt->shm
            vine_data_shm_sync(accel, "syncFrom", vdata, block);
            vdata->flags |= SHM_SYNC;
        case REMT_SYNC | SHM_SYNC:
        case SHM_SYNC: // shm->usr
            if (vdata->user) {
                memcpy(vdata->user, vine_data_deref(vdata), vdata->size);
                vdata->flags |= USER_SYNC;
            }
        case USER_SYNC | SHM_SYNC:
        case USER_SYNC:
        case ALL_SYNC:
            break;
        default: // GCOV_EXCL_START
            fprintf(stderr, "%s(%p) unexpected flags %lu!\n", __func__, data, vdata->flags);
            vine_assert(!"Uninitialized buffer");
            break;
        case NONE_SYNC:
            fprintf(stderr, "%s(%p) called with uninitialized buffer!\n", __func__, data);
            vine_assert(!"Uninitialized buffer");
            // GCOV_EXCL_STOP
    }

    vine_data_check_flags(data); // Ensure flags are consistent
} /* vine_data_sync_from_remote */

int vine_data_has_remote(vine_data *data)
{
    vine_assert_obj(data, VINE_TYPE_DATA);
    vine_data_s *vdata;

    vdata = (vine_data_s *) data;

    return !!(vdata->remote);
}

void vine_data_modified(vine_data *data, vine_data_flags_e where)
{
    vine_assert_obj(data, VINE_TYPE_DATA);
    vine_data_s *vdata;

    vdata        = (vine_data_s *) data;
    vdata->flags = where;
}

#undef vine_data_stat

void vine_data_stat(vine_data *data, const char *file, size_t line)
{
    vine_assert_obj(data, VINE_TYPE_DATA);
    vine_data_s *vdata;

    vdata = (vine_data_s *) data;

    file += strlen(file);
    while (*file != '/')
        file--;

    int scsum = 0;
    int ucsum = 0;
    int cnt;
    char *bytes = vine_data_deref(data);

    for (cnt = 0; cnt < vine_data_size(data); cnt++) {
        scsum += *bytes;
        bytes++;
    }

    bytes = vdata->user;

    if (bytes) {
        for (cnt = 0; cnt < vine_data_size(data); cnt++) {
            ucsum += *bytes;
            bytes++;
        }
    }

    fprintf(stderr, "%s(%p)[%lu]:Flags(%s%s%s) %08x %08x ?????? @%lu:%s\n", __func__, vdata, vine_data_size(vdata),
      (vdata->flags & USER_SYNC) ? "U" : " ",
      (vdata->flags & SHM_SYNC) ? "S" : " ",
      (vdata->flags & REMT_SYNC) ? "R" : " ",
      ucsum,
      scsum,
      line, file
    );
} /* vine_data_stat */

VINE_OBJ_DTOR_DECL(vine_data_s)
{
    vine_assert_obj(obj, VINE_TYPE_DATA);
    vine_data_s *data = (vine_data_s *) obj;

    VINE_THROTTLE_DEBUG_PRINT("%s(%p) - START\n", __func__, data);

    if (data->remote) {
        if (!data->accel) {
            fprintf(stderr, "vine_data(%p) dtor called, with dangling remote, with no accel!\n", data);
            vine_assert(!"Orphan dangling remote");
        } else {
            void *args[4] =
            { data, data->remote, (void *) (size_t) data->size, ((vine_vaccel_s *) (data->accel))->phys };
            VINE_THROTTLE_DEBUG_PRINT("Atempt to free %p %p size:%lu\n", data, data->remote, vine_data_size(data));
            vine_proc_s *free = vine_proc_get("free");
            vine_task_issue(data->accel, free, args, sizeof(args), 0, 0, 0, 0); // &dtrdata
            vine_object_ref_dec(((vine_object_s *) (data->accel)));
        }
    } else {
        if (data->accel) {
            vine_assert(((vine_object_s *) (data->accel))->type == VINE_TYPE_VIRT_ACCEL);
            vine_object_ref_dec(((vine_object_s *) (data->accel)));
        } else {
            fprintf(stderr, "vine_data(%p,%s,size:%lu) dtor called, data possibly unused!\n", data, obj->name, vine_data_size(
                  data));
        }
    }

    obj->type = VINE_TYPE_COUNT;

    VINE_THROTTLE_DEBUG_PRINT("%s(%p) - END\n", __func__, data);
}
