#include "config.h"
#include "system.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <pwd.h>

void utils_config_write_long(const char *key,long value)
{
	FILE *conf = 0;
	char *err  = "";
	char path[896];
	err = system_home_path();
	if (!err) {
		err = "Could not find home path!";
		return;
	}

	snprintf(path, sizeof(path), "%s/.vinetalk", err);
	conf = fopen(path, "rw");
	fprintf(conf,"%s %ld\n",key,value);
	fclose(conf);
}

void utils_config_write_str(const char *key,char * value)
{
	FILE *conf = 0;
	char *err  = "";
	char path[896];
	err = system_home_path();
	if (!err) {
		err = "Could not find home path!";
		return;
	}

	snprintf(path, sizeof(path), "%s/.vinetalk", err);
	conf = fopen(path, "rw");
	fprintf(conf,"%s %s\n",key,value);
	fclose(conf);
}

int _utils_config_get_str(const char *key, char *value, size_t value_size)
{
	FILE *conf = 0;
	char *path = "";
	char ckey[128];
	char cval[896];
	int  line = 0;

	path = system_home_path();
	if (!path)
		return 0;

	snprintf(cval, sizeof(cval), "%s/.vinetalk", path);
	conf = fopen(cval, "r");

	if (!conf)
		return 0;

	while (++line) {
		if (fscanf(conf, "%s %s", ckey, cval) < 1) {
			return 0;
		}
		if ( !strncmp( ckey, key, sizeof(ckey) ) ) {
			/* Found the key i was looking for */
			strncpy(value, cval, value_size);
			fclose(conf);
			return strlen(cval);
		}
	}
	return 0;
}

int utils_config_get_str(const char *key, char *value, size_t value_size, char * def_val)
{
	if(!_utils_config_get_str(key,value,value_size))
	{
		if(def_val)
		{	// Not found, but have default, update with default
			utils_config_write_str(key,def_val);
			strncpy(value,def_val,value_size);
		}
		else
			fprintf(stderr, "Could not locate %s config string\n", key);
		return 0;
	}
	return 1;
}

int utils_config_get_bool(const char *key, int *value, int def_val)
{
	if ( utils_config_get_int(key, value, def_val) )
		if (*value == 0 || *value == 1)
			return 1;



	*value = def_val;
	return 0;
}

int utils_config_get_int(const char *key, int *value, int def_val)
{
	long cval;

	if ( utils_config_get_long(key, &cval, def_val) )
		if (INT_MAX >= cval && INT_MIN <= cval) {
			*value = cval;
			return 1; /* Value was an int */
		}



	*value = def_val;
	return 0;
}

int utils_config_get_long(const char *key, long *value, long def_val)
{
	char cval[22];
	char * end;
	if ( _utils_config_get_str( key, cval, sizeof(cval) ) ) {
		/* Key exists */
		errno = 0;
		*value  = strtol(cval, &end, 0);
		if (errno || end == cval) {
			utils_config_write_long(key,def_val);
			*value = def_val;
			return 0;
		}
		return 1;
	}
	*value = def_val;
	return 0;
}

int utils_config_get_size(const char *key, size_t *value, size_t def_val)
{
	long cval;

	if ( utils_config_get_long(key, &cval, def_val) )
		if (SIZE_MAX >= cval && 0 <= cval) {
			*value = cval;
			return 1; /* Value was an size_t */
		}



	*value = def_val;
	return 0;
}
