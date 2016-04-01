all: all_c all_cpp

all_c:
	gcc -lrt -g -I../ ex1.c ../libvine_talk_shm.a -o ex1c
	gcc -lrt -g -I../ ex2.c ../libvine_talk_shm.a -o ex2c

all_cpp:
	g++ -lrt -g -I../ ex1.c ../libvine_talk_shm.a -o ex1cpp
	g++ -lrt -g -I../ ex2.c ../libvine_talk_shm.a -o ex2cpp
clean:
	rm ex1c ex2c ex1cpp ex2cpp
