# Vine Talk Versions, add yours
vine_talk_versions=libvine_talk_empty.a libvine_talk_shm.a

all: ${vine_talk_versions}

libvine_talk_%.a: profiler core dlmalloc
	cd $* && $(MAKE)
	$(AR) -cqs $@ $*/*.o profiler/*.o core/*.o dlmalloc/*.o

profiler core dlmalloc:
	cd $@ && $(MAKE)

clean:
	@find -name '*.o' -exec rm -v {} \;
	@find -name '*.a' -exec rm -v {} \;
	cd docs && $(MAKE) clean
	cd examples && $(MAKE) clean

edit:
	kate `find -name '*.h'` `find -name '*.c'` `find -name 'Makefile'` &>/dev/null &
.PHONY: profiler core dlmalloc clean edit
