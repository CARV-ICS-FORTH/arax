CFLAGS= -I.. -I../profiler -I../core -g
SRC:=$(wildcard *.c)
OBJ:=$(SRC:.c=.o)
DIR:=$(shell pwd)
TARGET=vine_talk_$(shell basename ${DIR}).a

${TARGET}: ${OBJ}

clean:
	@find -name '*.o' -exec rm -v {} \;
	@find -name '*.a' -exec rm -v {} \;
