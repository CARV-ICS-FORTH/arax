#include "vine_talk_types.h"
#include <strings.h>

struct vine_accel_type_map
{
    const char *      str;
    vine_accel_type_e type;
};

struct vine_accel_type_map types_map[VINE_ACCEL_TYPES] = {
    { "any",       ANY       },
    { "gpu",       GPU       },
    { "gpu_soft",  GPU_SOFT  },
    { "cpu",       CPU       },
    { "sda",       SDA       },
    { "nano_arm",  NANO_ARM  },
    { "nano_core", NANO_CORE },
    { "OpenCL",    OPEN_CL   },
    { "HIP",       HIP       }
};

int vine_accel_valid_type(vine_accel_type_e type)
{
    return type < VINE_ACCEL_TYPES;
}

const char* vine_accel_type_to_str(vine_accel_type_e type)
{
    if (vine_accel_valid_type(type))
        return types_map[type].str;

    return 0;
}

vine_accel_type_e vine_accel_type_from_str(const char *type)
{
    vine_accel_type_e cnt;

    if (!type)
        return VINE_ACCEL_TYPES;

    for (cnt = ANY; cnt < VINE_ACCEL_TYPES; cnt++) {
        if (!types_map[cnt].str)
            continue;
        if (!strcasecmp(type, types_map[cnt].str))
            break;
    }
    return cnt;
}
