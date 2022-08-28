#include "config.h"
#include "system.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include "utils/arax_assert.h"
#include <pwd.h>

void utils_config_write_long(char *path, const char *key, long value)
{
    FILE *conf = 0;

    conf = fopen(path, "rw");
    fprintf(conf, "%s %ld\n", key, value);
    fclose(conf);
}

void utils_config_write_str(char *path, const char *key, const char *value)
{
    FILE *conf = 0;

    conf = fopen(path, "a+");
    fprintf(conf, "%s %s\n", key, value);
    fclose(conf);
}

char* utils_config_alloc_path(const char *path)
{
    char temp[4096] = { 0 };
    char *tp        = temp;
    size_t size     = sizeof(temp);

    if (!path)
        return 0;

    do{
        if (!size)
            return 0;

        switch (*path) {
            case '~': {
                const char *home = system_home_path();
                size_t home_len  = strlen(home);
                arax_assert(size - home_len <= sizeof(temp)); // would have overflowed
                strncat(tp, home, size);
                tp   += home_len;
                size -= home_len;
                break;
            }
            default:
                *tp = *path;
                tp++;
                size--;
        }
    }while (*(path++)); // ensure \0 gets copied
    tp = malloc(strlen(temp) + 1);
    strcpy(tp, temp);
    return tp;
}

void utils_config_free_path(char *path)
{
    free(path);
}

int _utils_config_get_str(char *path, const char *key, char *value, size_t value_size)
{
    FILE *conf = 0;
    char ckey[128];
    char cval[896];
    int line = 0;
    int len  = 0;

    conf = fopen(path, "r");

    if (!conf)
        return 0;

    while (++line) {
        if (fscanf(conf, "%127s %895s", ckey, cval) < 1) {
            break;
        }
        if (!strncmp(ckey, key, sizeof(ckey) ) ) {
            /* Found the key i was looking for */
            strncpy(value, cval, value_size);
            len = strlen(cval);
            break;
        }
    }
    fclose(conf);
    return len;
}

int utils_config_get_str(char *path, const char *key, char *value, size_t value_size, const char *def_val)
{
    if (!_utils_config_get_str(path, key, value, value_size)) {
        if (def_val) { // Not found, but have default, update with default
            utils_config_write_str(path, key, def_val);
            strncpy(value, def_val, value_size);
        } else {
            fprintf(stderr, "No default value for \'%s\' config key\n", key);
        }
        return 0;
    }
    return 1;
}

int utils_config_get_bool(char *path, const char *key, int *value, int def_val)
{
    if (utils_config_get_int(path, key, value, def_val) ) {
        if (*value == 0 || *value == 1)
            return 1;
    }


    *value = def_val;
    return 0;
}

int utils_config_get_int(char *path, const char *key, int *value, int def_val)
{
    long cval;

    if (utils_config_get_long(path, key, &cval, def_val) ) {
        if (INT_MAX >= cval && INT_MIN <= cval) {
            *value = cval;
            return 1; /* Value was an int */
        }
    }


    *value = def_val;
    return 0;
}

int utils_config_get_long(char *path, const char *key, long *value, long def_val)
{
    char cval[22];
    char *end;

    if (_utils_config_get_str(path, key, cval, sizeof(cval) ) ) {
        /* Key exists */
        errno  = 0;
        *value = strtol(cval, &end, 0);
        if (errno || end == cval) {
            utils_config_write_long(path, key, def_val);
            *value = def_val;
            return 0;
        }
        return 1;
    }
    *value = def_val;
    return 0;
}

int utils_config_get_size(char *path, const char *key, size_t *value, size_t def_val)
{
    long cval;

    if (utils_config_get_long(path, key, &cval, def_val) ) {
        if (SIZE_MAX >= cval && 0 <= cval) {
            *value = cval;
            return 1; /* Value was an size_t */
        }
    }


    *value = def_val;
    return 0;
}
