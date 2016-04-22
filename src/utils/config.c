#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pwd.h>

char* get_home_path()
{
	uid_t         uid = getuid();
	struct passwd *pw = getpwuid(uid);

	if (!pw)
		return 0;

	return pw->pw_dir;
}

int util_config_get_str(const char *key, char *value, size_t value_size)
{
	FILE *conf;
	char *err = "";
	char ckey[128];
	char cval[896];
	int  line = 0;

	err = get_home_path();
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
			return strlen(cval);
		}
	}
FAIL: snprintf(cval, sizeof(cval), "%s/.vinetalk", err);
	fprintf(stderr, "Could not locate %s at %s\n", key, cval);
	fprintf(stderr, "%s:%s\n", __func__, err);
	return 0;
}

int util_config_get_bool(const char *key, int *val,int def_val)
{
	if ( util_config_get_str(key, (char*)val, sizeof(*val)) ) {
		*val = ( ( *( (char*)val ) ) == '1' );
		return 1;
	}
	*val = def_val;
	return 0;
}

int util_config_get_int(const char *key, int *val,int def_val)
{
	char cval[16];
	if ( util_config_get_str(key, (char*)cval, sizeof(cval)) ) {
		*val = atoi(cval);
		return 1;
	}
	*val = def_val;
	return 0;
}
