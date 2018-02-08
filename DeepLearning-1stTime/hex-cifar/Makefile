LIBGD=../SFMT-src-1.5.1

all: cifar_test bp_cifar bp_4stacks_cifar bp_5stacks_cifar bp_6stacks_cifar bp_7stacks_cifar bp_8stacks_cifar bp_9stacks_cifar bp_10stacks_cifar bp_cifar_hid bp_cifar_bat bp_cifar_epoch

bp_4stacks_cifar: bp_4stacks_cifar.c $(LIBGD)/SFMT.c cifar.c
	gcc -O3 -std=c99 -Wall -I $(LIBGD) -D SFMT_MEXP=19937 -I $(LIBGD)/include -L $(LIBGD)/lib -o $@ $^ -lgd -lm

bp_5stacks_cifar: bp_5stacks_cifar.c $(LIBGD)/SFMT.c cifar.c
	gcc -O3 -std=c99 -Wall -I $(LIBGD) -D SFMT_MEXP=19937 -I $(LIBGD)/include -L $(LIBGD)/lib -o $@ $^ -lgd -lm

bp_6stacks_cifar: bp_6stacks_cifar.c $(LIBGD)/SFMT.c cifar.c
	gcc -O3 -std=c99 -Wall -I $(LIBGD) -D SFMT_MEXP=19937 -I $(LIBGD)/include -L $(LIBGD)/lib -o $@ $^ -lgd -lm

bp_7stacks_cifar: bp_7stacks_cifar.c $(LIBGD)/SFMT.c cifar.c
	gcc -O3 -std=c99 -Wall -I $(LIBGD) -D SFMT_MEXP=19937 -I $(LIBGD)/include -L $(LIBGD)/lib -o $@ $^ -lgd -lm

bp_8stacks_cifar: bp_8stacks_cifar.c $(LIBGD)/SFMT.c cifar.c
	gcc -O3 -std=c99 -Wall -I $(LIBGD) -D SFMT_MEXP=19937 -I $(LIBGD)/include -L $(LIBGD)/lib -o $@ $^ -lgd -lm

bp_9stacks_cifar: bp_9stacks_cifar.c $(LIBGD)/SFMT.c cifar.c
	gcc -O3 -std=c99 -Wall -I $(LIBGD) -D SFMT_MEXP=19937 -I $(LIBGD)/include -L $(LIBGD)/lib -o $@ $^ -lgd -lm

bp_10stacks_cifar: bp_10stacks_cifar.c $(LIBGD)/SFMT.c cifar.c
	gcc -O3 -std=c99 -Wall -I $(LIBGD) -D SFMT_MEXP=19937 -I $(LIBGD)/include -L $(LIBGD)/lib -o $@ $^ -lgd -lm

bp_cifar: bp_cifar.c $(LIBGD)/SFMT.c cifar.c
	gcc -O3 -std=c99 -Wall -I $(LIBGD) -D SFMT_MEXP=19937 -I $(LIBGD)/include -L $(LIBGD)/lib -o $@ $^ -lgd -lm

bp_cifar_hid: bp_cifar_hid.c $(LIBGD)/SFMT.c cifar.c
	gcc -O3 -std=c99 -Wall -I $(LIBGD) -D SFMT_MEXP=19937 -I $(LIBGD)/include -L $(LIBGD)/lib -o $@ $^ -lgd -lm

bp_cifar_bat: bp_cifar_bat.c $(LIBGD)/SFMT.c cifar.c
	gcc -O3 -std=c99 -Wall -I $(LIBGD) -D SFMT_MEXP=19937 -I $(LIBGD)/include -L $(LIBGD)/lib -o $@ $^ -lgd -lm

bp_cifar_epoch: bp_cifar_epoch.c $(LIBGD)/SFMT.c cifar.c
	gcc -O3 -std=c99 -Wall -I $(LIBGD) -D SFMT_MEXP=19937 -I $(LIBGD)/include -L $(LIBGD)/lib -o $@ $^ -lgd -lm

cifar_test: cifar_test.c cifar.c
	gcc -O3 -std=c99 -Wall -I $(LIBGD)/include -L $(LIBGD)/lib -o $@ $^ -lgd

clean:
	rm -f *.o cifar_test bp_cifar bp_4stacks_cifar bp_5stacks_cifar bp_6stacks_cifar bp_7stacks_cifar bp_8stacks_cifar bp_9stacks_cifar bp_10stacks_cifar bp_cifar_hid bp_cifar_bat bp_cifar_epoch
