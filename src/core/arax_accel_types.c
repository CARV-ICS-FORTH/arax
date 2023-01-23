#include "arax_types.h"
#include <strings.h>

struct arax_accel_type_map
{
    const char *      str;
    arax_accel_type_e type;
};

struct arax_accel_type_map types_map[ARAX_ACCEL_TYPES] = {
    { "any",       ANY       },
    { "gpu",       GPU       },
    { "gpu_soft",  GPU_SOFT  },
    { "cpu",       CPU       },
    { "sda",       SDA       },
    { "nano_arm",  NANO_ARM  },
    { "nano_core", NANO_CORE },
    { "Open_CL",   OPEN_CL   },
    { "HIP",       HIP       }
};

int arax_accel_valid_type(arax_accel_type_e type)
{
    return type < ARAX_ACCEL_TYPES;
}

const char* arax_accel_type_to_str(arax_accel_type_e type)
{
    if (arax_accel_valid_type(type))
        return types_map[type].str;

    return 0;
}

arax_accel_type_e arax_accel_type_from_str(const char *type)
{
    arax_accel_type_e cnt;

    if (!type)
        return ARAX_ACCEL_TYPES;

    for (cnt = ANY; cnt < ARAX_ACCEL_TYPES; cnt++) {
        if (!types_map[cnt].str)
            continue;
        if (!strcasecmp(type, types_map[cnt].str))
            break;
    }
    return cnt;
}
