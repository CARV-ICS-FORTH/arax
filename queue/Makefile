all: test

test: test.c atring.c
	gcc -lpthread -lrt -g $^ -o test

clean:
	-rm test
