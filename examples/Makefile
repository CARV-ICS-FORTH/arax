all: all_c all_cpp

all_c:
	gcc -lrt -g -I../include ex1.c ../build/libvine.a -o ex1c
	gcc -lrt -g -I../include -I../src ex2.c ../build/libvine.a -o ex2c

all_cpp:
	g++ -lrt -g -I../include ex1.c ../build/libvine.a -o ex1cpp
	g++ -lrt -g -I../include -I../src  ex2.c ../build/libvine.a -o ex2cpp
clean:
	rm ex1c ex2c ex1cpp ex2cpp
