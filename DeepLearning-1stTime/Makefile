SFMT = ./SFMT-src-1.5.1

all: skel perceptron bp mnist_test bp_mnist bp_stack_mnist bp_mnist_epoch bp_4stacks_mnist bp_5stacks_mnist bp_6stacks_mnist bp_7stacks_mnist bp_8stacks_mnist bp_9stacks_mnist bp_10stacks_mnist bp_relu_mnist bp_mnist_bat

skel: skel.c $(SFMT)/SFMT.c
	gcc -O3 -std=c99 -Wall -I $(SFMT) -D SFMT_MEXP=19937 -o $@ $^ -lm

perceptron: perceptron.c $(SFMT)/SFMT.c
	gcc -O3 -std=c99 -Wall -I $(SFMT) -D SFMT_MEXP=19937 -o $@ $^ -lm

bp: bp.c $(SFMT)/SFMT.c
	gcc -O3 -std=c99 -Wall -I $(SFMT) -D SFMT_MEXP=19937 -o $@ $^ -lm

bp_mnist: bp_mnist.c $(SFMT)/SFMT.c mnist.c
	gcc -O3 -std=c99 -Wall -I $(SFMT) -D SFMT_MEXP=19937 -o $@ $^ -lgd -lm

bp_relu_mnist: bp_relu_mnist.c $(SFMT)/SFMT.c mnist.c
	gcc -O3 -std=c99 -Wall -I $(SFMT) -D SFMT_MEXP=19937 -o $@ $^ -lgd -lm

bp_mnist_epoch: bp_mnist_epoch.c $(SFMT)/SFMT.c mnist.c
	gcc -O3 -std=c99 -Wall -I $(SFMT) -D SFMT_MEXP=19937 -o $@ $^ -lgd -lm

bp_mnist_bat: bp_mnist_bat.c $(SFMT)/SFMT.c mnist.c
	gcc -O3 -std=c99 -Wall -I $(SFMT) -D SFMT_MEXP=19937 -o $@ $^ -lgd -lm

bp_stack_mnist: bp_stack_mnist.c $(SFMT)/SFMT.c mnist.c
	gcc -O3 -std=c99 -Wall -I $(SFMT) -D SFMT_MEXP=19937 -o $@ $^ -lgd -lm

bp_4stacks_mnist: bp_4stacks_mnist.c $(SFMT)/SFMT.c mnist.c
	gcc -O3 -std=c99 -Wall -I $(SFMT) -D SFMT_MEXP=19937 -o $@ $^ -lgd -lm

bp_5stacks_mnist: bp_5stacks_mnist.c $(SFMT)/SFMT.c mnist.c
	gcc -O3 -std=c99 -Wall -I $(SFMT) -D SFMT_MEXP=19937 -o $@ $^ -lgd -lm

bp_6stacks_mnist: bp_6stacks_mnist.c $(SFMT)/SFMT.c mnist.c
	gcc -O3 -std=c99 -Wall -I $(SFMT) -D SFMT_MEXP=19937 -o $@ $^ -lgd -lm

bp_7stacks_mnist: bp_7stacks_mnist.c $(SFMT)/SFMT.c mnist.c
	gcc -O3 -std=c99 -Wall -I $(SFMT) -D SFMT_MEXP=19937 -o $@ $^ -lgd -lm

bp_8stacks_mnist: bp_8stacks_mnist.c $(SFMT)/SFMT.c mnist.c
	gcc -O3 -std=c99 -Wall -I $(SFMT) -D SFMT_MEXP=19937 -o $@ $^ -lgd -lm

bp_9stacks_mnist: bp_9stacks_mnist.c $(SFMT)/SFMT.c mnist.c
	gcc -O3 -std=c99 -Wall -I $(SFMT) -D SFMT_MEXP=19937 -o $@ $^ -lgd -lm

bp_10stacks_mnist: bp_10stacks_mnist.c $(SFMT)/SFMT.c mnist.c
	gcc -O3 -std=c99 -Wall -I $(SFMT) -D SFMT_MEXP=19937 -o $@ $^ -lgd -lm

mnist_test: mnist_test.c mnist.c
	gcc -O3 -std=c99 -Wall -o $@ $^ -lgd

clean:
	rm -f *.o skel perceptron bp mnist_test bp_mnist bp_stack_mnist bp_mnist_epoch bp_4stacks_mnist bp_5stacks_mnist bp_6stacks_mnist bp_7stacks_mnist bp_8stacks_mnist bp_9stacks_mnist bp_10stacks_mnist bp_relu_mnist bp_mnist_bat
