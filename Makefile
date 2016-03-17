# Vine Talk Versions, add yours
HDR_INCS=-I./ -I./profiler
LIB_INCS=-L./profiler -lprofiler

vine_talk_versions=libvine_talk_empty.a libvine_talk_local_profile.a

test=$(shell echo $(1))

all: ${vine_talk_versions}

%.a: %.o libprofiler.o
	ar -cvq $@ $^
OBJS=

libvine_talk_%.o: %/*.c
	cat $^ | gcc ${HDR_INCS} ${LIB_INCS} -c $^ -o $@

lib%.o: %/*.c
	cat $^ | gcc ${HDR_INCS} ${LIB_INCS} -c -x c - -o $@


clean:
	@find -name '*.o' -exec rm -v {} \;
	@find -name '*.a' -exec rm -v {} \;

