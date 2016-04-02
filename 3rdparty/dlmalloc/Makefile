CFLAGS=-DONLY_MSPACES -DUSE_SPIN_LOCKS  -DMSPACES -g

all:  malloc.o

test: malloc.o test.o
	gcc $^ -o test

clean:
	@find -name '*.o' -exec rm -v {} \;
	@find -name 'test' -exec rm -v {} \;
