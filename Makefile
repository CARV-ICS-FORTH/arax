# Vine Talk Versions, add yours
vine_talk_versions=libvine_talk_noop.a libvine_talk_local_profile.a

all: ${vine_talk_versions}

%.a: %.o
	ar -cvq $@ $<

libvine_talk_%.o: %/*.c
	gcc -I. -c $^ -o $@

clean:
	find -name '*.o' -delete
	find -name '*.a' -delete
