# Vine Talk Versions, add yours
vine_talk_versions=libvine_talk_empty.a libvine_talk_local_profile.a

all: ${vine_talk_versions}

%.a: %.o
	ar -cvq $@ $^

libvine_talk_%.a: profiler/libprofiler.a
	cd $* && make
	ar -cvq $@ $*/libvine_talk_$*.a $<

profiler/libprofiler.a:
	cd profiler && make

clean:
	@find -name '*.o' -exec rm -v {} \;
	@find -name '*.a' -exec rm -v {} \;
	cd docs && make clean
