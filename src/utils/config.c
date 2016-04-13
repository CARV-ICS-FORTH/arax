#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pwd.h>

char * get_home_path()
{
	uid_t uid = getuid();
	struct passwd *pw = getpwuid(uid);

	if (!pw)
		return 0;

	return pw->pw_dir;
}

int util_config_get(const char * key,char * value,size_t value_size)
{
	FILE * conf;
	char * err = "";
	char ckey[1024];
	int line = 0;

	err = get_home_path();
	if(!err)
	{
		err = "Could not find home path!";
		goto FAIL;
	}

	sprintf(ckey,"%s/.vinetalk",err);
	conf = fopen(ckey,"r");

	if(!conf)
	{
		err = "Could not open ~/.vinetalk!";
		goto FAIL;
	}

	while(++line)
	{
		if(fscanf(conf,"%s",ckey) < 1)
		{
			err = "Reched EOF";
			goto FAIL;
		}
		fgets(value,value_size,conf);
		value[strlen(value)-1] = 0;
		if(strcmp(key,ckey))
			continue;
		return strlen(value);
	}

	FAIL:
	sprintf(ckey,"%s/.vinetalk",err);
	fprintf(stderr,"Could not locate %s at %s\n",key,ckey);
	fprintf(stderr,"%s:%s\n",__func__,err);
	return 0;
}
