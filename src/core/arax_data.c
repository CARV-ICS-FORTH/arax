#include "arax_data.h"
#include "arax_task.h"
#include "arax_data_private.h"
#include "arax_pipe.h"
#include "arax_ptr.h"
#include <string.h>
#include <stdlib.h>
#include "utils/system.h"

// #define printd(...) fprintf(__VA_ARGS__)
#define  printd(...)

#define VDFLAG(DATA, FLAG)  (DATA->flags & FLAG)// ((size_t*)BUFF-1) same pointer arithmetics//
#define VD_BUFF_OWNER(BUFF) *(arax_data_s **) ((char *) BUFF - sizeof(size_t *))

arax_data_s* arax_data_init(arax_pipe_s *vpipe, size_t size)
{
    return arax_data_init_aligned(vpipe, size, 1);
} /* arax_data_init */

arax_data_s* arax_data_init_aligned(arax_pipe_s *vpipe, size_t size, size_t align)
{
    arax_data_s *data;
    size_t alloc_size = sizeof(arax_data_s) + ARAX_BUFF_ALLOC_SIZE(size, align);

    arax_assert(align);

    data = (arax_data_s *) arax_object_register(&(vpipe->objs),
        ARAX_TYPE_DATA,
        "UNUSED", alloc_size, 1);

    if (!data)     // GCOV_EXCL_LINE
        return 0;  // GCOV_EXCL_LINE

    uint64_t *buff_ptr = (uint64_t *) (data + 1); // Skip the struct

    buff_ptr++; // Skip a back_pointer
    buff_ptr = (uint64_t *) (((char *) buff_ptr) + align - (((size_t) buff_ptr) % align));// Align ptr;
    arax_data_s **back_pointer = (arax_data_s **) (buff_ptr - 1);

    *back_pointer = data;

    data->size   = size;
    data->buffer = buff_ptr;
    data->align  = align;
    data->flags  = 0;
    data->phys   = 0; // required for migration

    return data;
} /* arax_data_init_aligned */

void arax_data_get(arax_data *data, void *user)
{
    arax_proc_s *get_proc = arax_proc_get(__func__);

    arax_assert_obj(data, ARAX_TYPE_DATA);
    arax_data_s *vd = (arax_data_s *) data;

    arax_assert(vd->accel);

    // We must wait all previous operations to complete to ensure we get
    // up to date data.Also have to synchronize data up to shm.

    arax_task_msg_s *task = arax_task_issue(vd->accel, get_proc, 0, arax_data_size(data), 0, 0, 1, &data);

    arax_assert(arax_task_wait(task) == task_completed);

    memcpy(user, arax_task_host_data(task, arax_data_size(vd)), arax_data_size(vd));

    arax_task_free(task);

    arax_proc_put(get_proc);
}

void arax_data_set(arax_data *data, arax_accel *accel, const void *user)
{
    arax_proc_s *set_proc = arax_proc_get(__func__);

    arax_assert_obj(data, ARAX_TYPE_DATA);
    arax_data_s *vd = (arax_data_s *) data;

    // If already submitted to a vac, it should be at the same
    arax_assert( (!(vd->accel)) || (vd->accel == accel) );
    if (vd->accel == 0)
        arax_object_ref_inc(accel);
    vd->accel = accel;

    size_t size = arax_data_size(vd);

    arax_task_issue(accel, set_proc, (void *) user, size, 0, 0, 1, &data);

    arax_proc_put(set_proc);
}

void arax_data_check_flags(arax_data_s *data)
{
    switch (data->flags & ALL_SYNC) {
        case NONE_SYNC:
        case SHM_SYNC:
        case REMT_SYNC:
        case ALL_SYNC:
            return;

        default: // GCOV_EXCL_START
            fprintf(stderr, "%s(%p): Inconsistent data flags %lu\n", __func__, data, data->flags);
            arax_assert(!"Inconsistent data flags");
            // GCOV_EXCL_STOP
    }
}

void arax_data_memcpy(arax_accel *accel, arax_data_s *dst, arax_data_s *src, int block)
{
    arax_assert_obj(dst, ARAX_TYPE_DATA);
    arax_assert_obj(src, ARAX_TYPE_DATA);

    if (dst == src)
        return;

    if (arax_data_size(dst) != arax_data_size(src)) {
        fprintf(stderr, "%s(%p,%p): Size mismatch (%lu,%lu)\n", __func__, dst, src, arax_data_size(dst),
          arax_data_size(src));
        arax_assert(!"Size mismatch");
    }
    fprintf(stderr, "%s(%p,%p)[%lu,%lu]\n", __func__, dst, src, dst->flags, src->flags);

    arax_data_get(src, 0);

    arax_data_set(dst, accel, arax_data_deref(src));
}

#define TYPE_MASK(A, B) ( ( (A) *ARAX_TYPE_COUNT ) + (B) )

void arax_data_migrate_accel(arax_data_s *data, arax_accel *accel)
{
    arax_assert_obj(data, ARAX_TYPE_DATA);
    arax_assert(accel); // Must be given a valid accelerator

    if (data->accel == accel) // Already assigned to accel - nop
        return;

    if (!data->accel) {             // Unassinged data - assign and increase accel
        arax_assert(!data->remote); // How can you have a remote ptr, with no accel?
        arax_object_ref_inc(((arax_object_s *) accel));
        data->accel = accel;
        return;
    }

    arax_object_type_e data_accel_type  = ((arax_object_s *) (data->accel))->type;
    arax_object_type_e accel_accel_type = ((arax_object_s *) (accel))->type;

    switch (TYPE_MASK(data_accel_type, accel_accel_type) ) {
        case TYPE_MASK(ARAX_TYPE_VIRT_ACCEL, ARAX_TYPE_VIRT_ACCEL): {
            arax_vaccel_s *dvac = (arax_vaccel_s *) (data->accel);
            arax_vaccel_s *avac = (arax_vaccel_s *) (accel);
            if (dvac->phys == avac->phys) { // Both use the same physical accel - just migrate ref counts
                arax_object_ref_dec(((arax_object_s *) dvac));
                arax_object_ref_inc(((arax_object_s *) avac));
                data->accel = accel;
                return;
            } else { // Device 2 Device migration - migrate data and ref counts
                fprintf(stderr, "Different device data migration!\n");
                arax_assert(!"arax_data_migrate_accel: D2D not implemented!");
            }
        }
        break;
        default: {
            fprintf(stderr, "%s():Data migration not implemented(%s:%s,%s:%s)!\n",
              __func__,
              arax_object_type_to_str(data_accel_type), ((arax_object_s *) (data->accel))->name,
              arax_object_type_to_str(accel_accel_type), ((arax_object_s *) (accel))->name
            );
            arax_assert(!"No migration possible");
        }
        break;
    }
} /* arax_data_migrate_accel */

void arax_data_allocate_remote(arax_data_s *data, arax_accel *accel)
{
    arax_assert_obj(data, ARAX_TYPE_DATA);
    arax_assert_obj(accel, ARAX_TYPE_VIRT_ACCEL);

    if ( ((arax_vaccel_s *) accel)->type == CPU)
        return;  // CPU does not have a 'remote', so nothing to do

    if (data->remote) {
        arax_assert(data->accel); // Allocate remote must have accel
        return;                   // Nothing left to do
    }

    if (!accel) {
        arax_object_ref_inc((arax_object_s *) accel);
        data->accel = accel;
    }

    ARAX_THROTTLE_DEBUG_PRINT("%s(%p) - start\n", __func__, data);

    arax_proc_s *alloc_data = arax_proc_get("alloc_data");
    arax_task_msg_s *task   = arax_task_issue(accel, alloc_data, 0, 0, 0, 0, 1, (arax_data **) &data);

    arax_assert(arax_task_wait(task) == task_completed);
    arax_task_free(task);
    arax_assert(data->remote); // Ensure remote was allocated

    arax_assert(data->accel == accel);

    arax_accel_size_dec(((arax_vaccel_s *) accel)->phys, arax_data_size(data));
    ARAX_THROTTLE_DEBUG_PRINT("%s(%p) - end\n", __func__, data);
}

void arax_data_set_accel(arax_data_s *data, arax_accel *accel)
{
    arax_assert_obj(data, ARAX_TYPE_DATA);
    arax_assert_obj(accel, ARAX_TYPE_VIRT_ACCEL);

    if (data->accel != accel) {
        data->accel = accel;
        arax_object_ref_inc(accel);
    }
}

void arax_data_set_remote(arax_data_s *data, arax_accel *accel, void *remt)
{
    arax_assert_obj(data, ARAX_TYPE_DATA);
    arax_assert_obj(accel, ARAX_TYPE_VIRT_ACCEL);
    arax_assert(((arax_vaccel_s *) accel)->type != CPU);
    arax_assert(data->accel == 0);
    arax_assert(data->remote == 0);

    arax_object_ref_inc((arax_object_s *) accel);
    data->accel  = accel;
    data->remote = remt;
    data->flags |= OTHR_REMT;
}

void arax_data_arg_init(arax_data_s *data, arax_accel *accel)
{
    // check errors
    arax_assert_obj(data, ARAX_TYPE_DATA);
    arax_assert(accel);

    arax_data_migrate_accel(data, accel);
}

void arax_data_input_init(arax_data_s *data, arax_accel *accel)
{
    // check errors
    arax_assert_obj(data, ARAX_TYPE_DATA);
    arax_assert(accel);

    arax_object_ref_inc(&(data->obj));

    arax_data_migrate_accel(data, accel);
}

void arax_data_output_init(arax_data_s *data, arax_accel *accel)
{
    // check errors
    arax_assert_obj(data, ARAX_TYPE_DATA);
    arax_assert(accel);

    arax_object_ref_inc(&(data->obj));

    arax_data_migrate_accel(data, accel);
}

size_t arax_data_size(arax_data *data)
{
    arax_assert_obj(data, ARAX_TYPE_DATA);
    arax_data_s *vdata;

    vdata = data;
    return vdata->size;
}

void* arax_data_deref(arax_data *data)
{
    arax_assert_obj(data, ARAX_TYPE_DATA);
    arax_data_s *vdata;

    vdata = (arax_data_s *) data;

    ARAX_THROTTLE_DEBUG_PRINT("%s(%p) - start\n", __func__, data);

    ARAX_THROTTLE_DEBUG_PRINT("%s(%p) - end\n", __func__, data);

    return vdata->buffer;
}

arax_data* arax_data_ref(void *data)
{
    arax_assert(data);

    arax_data_s *vdata = VD_BUFF_OWNER(data);

    // GCOV_EXCL_START
    if (!vdata)
        return 0;

    if (!arax_ptr_valid(vdata))
        return 0;

    if (vdata->obj.type != ARAX_TYPE_DATA)
        return 0;

    // GCOV_EXCL_STOP

    return vdata;
}

arax_data* arax_data_ref_offset(arax_pipe_s *vpipe, void *data)
{
    arax_assert(data);
    arax_assert(arax_ptr_valid(data));
    // Might be at the start
    arax_data_s *ret = arax_data_ref(data);

    if (ret)
        return ret;

    utils_list_s *datas = arax_object_list_lock(&(vpipe->objs), ARAX_TYPE_DATA);
    utils_list_node_s *itr;

    utils_list_for_each(*datas, itr)
    {
        arax_data_s *vd = (arax_data_s *) (itr->owner);
        void *start     = arax_data_deref(vd);
        void *end       = start + arax_data_size(vd);

        if (data > start && data < end) {
            ret = vd;
            break; // Found it!
        }
    }

    arax_object_list_unlock(&(vpipe->objs), ARAX_TYPE_DATA);

    return ret;
}

void arax_data_free(arax_data *data)
{
    arax_assert_obj(data, ARAX_TYPE_DATA);
    arax_data_s *vdata;

    vdata = (arax_data_s *) data;
    arax_object_ref_dec(&(vdata->obj));
}

int arax_data_has_remote(arax_data *data)
{
    arax_assert_obj(data, ARAX_TYPE_DATA);
    arax_data_s *vdata;

    vdata = (arax_data_s *) data;

    return !!(vdata->remote);
}

void arax_data_modified(arax_data *data, arax_data_flags_e where)
{
    arax_assert_obj(data, ARAX_TYPE_DATA);
    arax_data_s *vdata;

    vdata        = (arax_data_s *) data;
    vdata->flags = (vdata->flags & OTHR_REMT) | where;
}

#undef arax_data_stat

void arax_data_stat(arax_data *data, const char *file, size_t line)
{
    arax_assert_obj(data, ARAX_TYPE_DATA);
    arax_data_s *vdata;

    vdata = (arax_data_s *) data;

    file += strlen(file);
    while (*file != '/')
        file--;

    int scsum = 0;
    int cnt;
    char *bytes = arax_data_deref(data);

    for (cnt = 0; cnt < arax_data_size(data); cnt++) {
        scsum += *bytes;
        bytes++;
    }

    fprintf(stderr, "%s(%p)[%lu]:Flags(%s%s%s) %08x ?????? @%lu:%s\n", __func__, vdata, arax_data_size(vdata),
      (vdata->flags & SHM_SYNC) ? "S" : " ",
      (vdata->flags & REMT_SYNC) ? "R" : " ",
      (vdata->flags & OTHR_REMT) ? "O" : " ",
      scsum,
      line, file
    );
} /* arax_data_stat */

ARAX_OBJ_DTOR_DECL(arax_data_s)
{
    arax_assert_obj(obj, ARAX_TYPE_DATA);
    arax_data_s *data = (arax_data_s *) obj;

    ARAX_THROTTLE_DEBUG_PRINT("%s(%p) - START\n", __func__, data);

    if (data->remote && ((data->flags & OTHR_REMT) == 0)) {
        if (!data->phys) {
            fprintf(stderr, "arax_data(%p) dtor called, with dangling remote, with no accel!\n", data);
            arax_assert(!"Orphan dangling remote");
        } else {
            void *args[4] =
            { data, data->remote, (void *) (size_t) data->size, (arax_vaccel_s *) data->phys };
            ARAX_THROTTLE_DEBUG_PRINT("Atempt to free %p %p size:%lu\n", data, data->remote, arax_data_size(data));
            arax_proc_s *free = arax_proc_get("free");
            arax_task_issue(data->phys->free_vaq, free, args, sizeof(args), 0, 0, 0, 0);
            arax_object_ref_dec(((arax_object_s *) (data->accel)));
        }
    } else {
        if (data->accel) {
            arax_assert(((arax_object_s *) (data->accel))->type == ARAX_TYPE_VIRT_ACCEL);
            arax_object_ref_dec(((arax_object_s *) (data->accel)));
        } else {
            fprintf(stderr, "arax_data(%p,%s,size:%lu) dtor called, data possibly unused!\n", data, obj->name, arax_data_size(
                  data));
        }
    }

    obj->type = ARAX_TYPE_COUNT;
}
