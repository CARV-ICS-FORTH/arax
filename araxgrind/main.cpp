#include "Args.h"
#include <map>
#include <conf.h>
#include <unistd.h>
#include <climits>
#include <iomanip>
#include <iostream>
#include <vector>
#include <arax_pipe.h>
#include <core/arax_data.h>
#include <core/arax_data_private.h>
#include <core/arax_ptr.h>

std::size_t getSizeOfAraxObject(arax_object_s & obj)
{
    switch (obj.type) {
        case ARAX_TYPE_PHYS_ACCEL: /* Physical Accelerator */
            return sizeof(arax_accel_s);

        case ARAX_TYPE_VIRT_ACCEL: /* Virtual Accelerator */
            return sizeof(arax_vaccel_s);

        case ARAX_TYPE_PROC: /* Procedure */
            return sizeof(arax_proc_s);

        case ARAX_TYPE_DATA: /* Data Allocation */
            return sizeof(arax_data_s) + arax_data_size((arax_data_s *) &obj);

        case ARAX_TYPE_TASK: /* Task */
            return sizeof(arax_task_msg_s);

        default:
            throw std::runtime_error("Encountered unknown object type (" + std::to_string(obj.type) + ")");
    }
    ;
}

std::string getNameOfAraxObject(arax_object_s & obj)
{
    if (obj.type == ARAX_TYPE_TASK) {
        arax_task_msg_s *task = (arax_task_msg_s *) &obj;

        if (!task->proc)
            return "NULL PROC TASK";

        return ((arax_proc_s *) (task->proc))->obj.name;
    } else {
        return obj.name;
    }
}

std::string printSize(std::size_t size)
{
    #define POWER_SIZE 6
    const char *unit[POWER_SIZE] =
    {
        " B",
        "KB",
        "MB",
        "GB",
        "TB",
        "PB"
    };
    std::size_t power = 0;
    while (size >= 1024 * 10 && power < POWER_SIZE) {
        size /= 1024;
        power++;
    }
    return std::to_string(size) + unit[power];
}

std::size_t max_name_len = 0;

class Leak
{
    friend std::ostream & operator << (std::ostream & os, const Leak & leak);
public:
    Leak()
        : leaks(0), size(0), total(0), max_ref(0), min_ref(INT_MAX)
    { }

    void track(arax_object_s & obj)
    {
        int refs = arax_object_refs(&obj);

        leaks++;
        name         = getNameOfAraxObject(obj);
        max_name_len = std::max(name.size(), max_name_len);
        size         = getSizeOfAraxObject(obj);
        instances.emplace_back(&obj, refs);
        total  += size;
        max_ref = std::max(max_ref, refs);
        min_ref = std::min(min_ref, refs);
    }

private:
    std::vector<std::pair<arax_object_s *, int> > instances;
    std::string name;
    uint64_t leaks;
    std::size_t size;
    std::size_t total;
    int max_ref;
    int min_ref;
};

std::ostream & operator << (std::ostream & os, const Leak & leak)
{
    os.width(9);
    os << leak.leaks << " leak" << ((leak.leaks > 1) ? "s" : " ") << " of ";
    os.width(9);
    os << printSize(leak.size) << " from ";
    os.width(max_name_len);
    os << leak.name << " (total: ";
    os.width(6);
    os << std::setprecision(5)
       << std::fixed << printSize(leak.total)
       << ", refs: [" << leak.min_ref << " , "
       << leak.max_ref << "] )";

    if (getPtr()) {
        os << std::endl;
        for (auto instance : leak.instances) {
            os.width(32);
            os << instance.first << " refs: " << instance.second << std::endl;
            #ifdef ARAX_DATA_TRACK
            if (getTrack() && instance.first->type == ARAX_TYPE_DATA) {
                arax_data_s *data = (arax_data_s *) (instance.first);
                os << "Allocation track:" << data->alloc_track << "\n\n";
            }
            #endif
        }
    } else {
        os << std::endl;
    }

    return os;
}

void leak_check(arax_pipe_s *vpipe, arax_object_type_e type, std::string stype)
{
    std::size_t leaks_cnt  = 0;
    std::size_t leak_total = 0;
    utils_list_s *list;
    utils_list_node_s *itr;
    arax_object_s *obj;
    std::map<std::size_t, std::map<std::string, Leak> > leaks;

    list = arax_object_list_lock(&(vpipe->objs), type);

    utils_list_for_each(*list, itr)
    {
        obj = (arax_object_s *) itr->owner;
        leaks[getSizeOfAraxObject(*obj)][getNameOfAraxObject(*obj)].track(*obj);
        leaks_cnt++;
        leak_total += getSizeOfAraxObject(*obj);
    }

    arax_object_list_unlock(&(vpipe->objs), type);

    std::cerr << "Found " << leaks_cnt << " " << stype << " leaks, totaling " << printSize(leak_total) << ", from "
              << leaks.size() << " sources:\n";

    for (auto & leak_sz : leaks) {
        for (auto leak_n : leak_sz.second) {
            std::cerr << leak_n.second << std::endl;
        }
    }
}

/**
 * Pointer might look like an object of certain type but this can be wrong.
 * Assuming the type, search the object repo for the object.
 */
arax_object_type_e getCertainType(arax_pipe_s *vpipe, void *ptr)
{
    arax_object_s *obj = (arax_object_s *) ptr;
    // Lets assume it is what it looks
    arax_object_type_e possible_type = obj->type;
    // Assume pointer is not what it seems
    arax_object_type_e type = ARAX_TYPE_COUNT;

    if (possible_type >= ARAX_TYPE_COUNT) { // Not a valid enum value
        return ARAX_TYPE_COUNT;
    }

    utils_list_s *list = arax_object_list_lock(&(vpipe->objs), possible_type);
    utils_list_node_s *itr;

    utils_list_for_each(*list, itr)
    {
        if (itr->owner == ptr) { // I found it, it was what it seemed
            type = possible_type;
            break;
        }
    }
    arax_object_list_unlock(&(vpipe->objs), possible_type);

    return type;
}

void ptrType(std::ostream & os, arax_pipe_s *vpipe, void *ptr, arax_object_type_e & type)
{
    type = ARAX_TYPE_COUNT;

    if (!arax_ptr_valid(ptr) ) {
        os << "not a arax pointer!";
        return;
    }
    arax_object_s *obj = (arax_object_s *) ptr;

    arax_object_type_e actual_type = getCertainType(vpipe, obj);

    switch (actual_type) {
        case ARAX_TYPE_PHYS_ACCEL ... ARAX_TYPE_TASK:
            os << arax_object_type_to_str(actual_type) << " " << getNameOfAraxObject(*obj);
            break;
        default: {
            arax_data_s *data =
              (arax_data_s *) arax_data_ref_offset(vpipe, ptr);
            if (!data) {
                os << "not object pointer or inside arax_data buffer";
            } else {
                std::size_t ptr_s = (std::size_t) ptr;
                std::size_t buf_s = (std::size_t) arax_data_deref(data);
                os << ptr_s - buf_s
                   << " bytes inside buffer of Arax Data " << data << '"'
                   << getNameOfAraxObject(*obj) << '"' << std::endl;
            }
            break;
        }
    }
} // ptrType

void inspectPointer(std::ostream & os, arax_pipe_s *vpipe, void *ptr)
{
    arax_object_type_e type;

    os << "\t" << ptr << " is ";
    ptrType(os, vpipe, ptr, type);
    os << std::endl;
}

int main(int argc, char *argv[])
{
    if (!parseArgs(std::cerr, argc, argv))
        return -1;

    if (getHelp()) {
        printArgsHelp(std::cerr);
        return 0;
    }

    arax_pipe_s *vpipe = arax_init();

    do{
        #ifndef ARAX_DATA_ANNOTATE
        std::cerr << "Warning: ARAX_DATA_ANNOTATE not enabled, leaks will be anonymous!\n";
        #endif

        if (getAll() ) {
            leak_check(vpipe, ARAX_TYPE_PHYS_ACCEL, "Phys Accel");
            leak_check(vpipe, ARAX_TYPE_VIRT_ACCEL, "Virt Accel");
        }

        leak_check(vpipe, ARAX_TYPE_DATA, "data");
        leak_check(vpipe, ARAX_TYPE_TASK, "task");

        std::set<void *> iptrs = getInspectPointers();

        if (iptrs.size() ) {
            std::cerr << "Inspecting " << iptrs.size() << " pointers:\n";
            for (auto ptr : iptrs)
                inspectPointer(std::cerr, vpipe, ptr);
        }

        if (getRefresh()) {
            usleep(250 * 1000);
            std::cerr << (char) 27 << "[2J";
        }
    }while (getRefresh());
    arax_exit();
    return 0;
} // main
