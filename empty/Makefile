CFLAGS= -I.. -I../profiler
SRC:=$(wildcard *.c)
OBJ:=$(SRC:.c=.o)
DIR:=$(shell pwd)
TARGET=libvine_talk_$(shell basename ${DIR}).a

${TARGET}: ${OBJ}
	ar -cvq $@ $^

clean:
	@find -name '*.o' -exec rm -v {} \;
	@find -name '*.a' -exec rm -v {} \;
