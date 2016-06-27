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

int util_config_get_str(const char *key, char *value, size_t value_size)
{
	FILE *conf = 0;
	char *err  = "";
	char ckey[128];
	char cval[896];
	int  line = 0;

	err = system_home_path();
	if (!err) {
		err = "Could not find home path!";
		goto FAIL;
	}

	snprintf(cval, sizeof(cval), "%s/.vinetalk", err);
	conf = fopen(cval, "r");

	if (!conf) {
		err = "Could not open ~/.vinetalk!";
		goto FAIL;
	}

	while (++line) {
		if (fscanf(conf, "%s %s", ckey, cval) < 1) {
			err = "Reched EOF";
			goto FAIL;
		}
		if ( !strncmp( ckey, key, sizeof(ckey) ) ) {
			/* Found the key i was looking for */
			strncpy(value, cval, value_size);
			fclose(conf);
			return strlen(cval);
		}
	}
FAIL:   if (conf)
		fclose(conf);
	snprintf( cval, sizeof(cval), "%s/.vinetalk", system_home_path() );
	fprintf(stderr, "Could not locate %s at %s:%s\n", key, cval, err);
	fprintf(stderr, "%s:%s\n", __func__, err);
	return 0;
}

int util_config_get_bool(const char *key, int *value, int def_val)
{
	if ( util_config_get_int(key, value, def_val) )
		if (*value == 0 || *value == 1)
			return 1;



	*value = def_val;
	return 0;
}

int util_config_get_int(const char *key, int *value, int def_val)
{
	long cval;

	if ( util_config_get_long(key, &cval, def_val) )
		if (INT_MAX >= cval && INT_MIN <= cval) {
			*value = cval;
			return 1; /* Value was an int */
		}



	*value = def_val;
	return 0;
}

int util_config_get_long(const char *key, long *value, long def_val)
{
	char cval[22];
	char * end;
	if ( util_config_get_str( key, cval, sizeof(cval) ) ) {
		/* Key exists */
		errno = 0;
		*value  = strtol(cval, &end, 0);
		if (errno || end == cval) {
			fprintf(stderr, "%s on key \"%s\"(%s)\n", strerror(
			                errno), key, cval);
			*value = def_val;
			return 0;
		}
		return 1;
	}
	*value = def_val;
	return 0;
}

int util_config_get_size(const char *key, size_t *value, size_t def_val)
{
	long cval;

	if ( util_config_get_long(key, &cval, def_val) )
		if (SIZE_MAX >= cval && 0 <= cval) {
			*value = cval;
			return 1; /* Value was an size_t */
		}



	*value = def_val;
	return 0;
}
