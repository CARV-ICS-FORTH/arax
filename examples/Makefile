lib_so=$(shell find ../ -name '*.a')
lib_versions:=$(lib_so:.a=)
lib_versions:=$(subst ../libvine_talk_,,${lib_versions})

all: ${lib_versions}

%: ../libvine_talk_%.a
	gcc -I.. -L.. -lvine_talk_$@ ex1.c $^ -o $@

clean:
	find -name '*.o' -delete
	rm ${lib_versions}

.IGNORE: $(lib_versions)
