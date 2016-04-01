CFLAGS= -I.. -I../profiler -I../../vine_pipe/alloc -I../../vine_pipe/queue
SRC:=$(wildcard *.c)
OBJ:=$(SRC:.c=.o)
DIR:=$(shell pwd)
TARGET=vine_talk_$(shell basename ${DIR}).a

${TARGET}: ${OBJ}

clean:
	@find -name '*.o' -exec rm -v {} \;
	@find -name '*.a' -exec rm -v {} \;
