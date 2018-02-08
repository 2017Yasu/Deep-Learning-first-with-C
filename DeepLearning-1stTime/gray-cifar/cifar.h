#ifndef _CIFAR_H_
#define _CIFAR_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gd.h>

#define CIFAR_TRAINING_BINARY_FILE1 "../cifar-10-batches-bin/data_batch_1.bin"
#define CIFAR_TRAINING_BINARY_FILE2 "../cifar-10-batches-bin/data_batch_2.bin"
#define CIFAR_TRAINING_BINARY_FILE3 "../cifar-10-batches-bin/data_batch_3.bin"
#define CIFAR_TRAINING_BINARY_FILE4 "../cifar-10-batches-bin/data_batch_4.bin"
#define CIFAR_TRAINING_BINARY_FILE5 "../cifar-10-batches-bin/data_batch_5.bin"
#define CIFAR_TEST_BINARY_FILE "../cifar-10-batches-bin/test_batch.bin"
#define CIFAR_LABEL_FILE "../cifar-10-batches-bin/batches.meta.txt"

#define CIFAR_TRAINING_DATA_SIZE 50000
#define CIFAR_TEST_DATA_SIZE 10000
#define CIFAR_IMAGE_ROW_SIZE 32
#define CIFAR_IMAGE_COL_SIZE 32
#define CIFAR_IMAGE_CHANNEL 3
#define CIFAR_IMAGE_PIXEL CIFAR_IMAGE_ROW_SIZE * CIFAR_IMAGE_COL_SIZE
#define CIFAR_IMAGE_SIZE CIFAR_IMAGE_ROW_SIZE * CIFAR_IMAGE_COL_SIZE * CIFAR_IMAGE_CHANNEL
#define CIFAR_LABEL_SIZE 10

void cifar_initialize ( double ***, int **, double ***, int **, char *** );
void cifar_finalize ( double **, int *, double **, int *, char ** );
void cifar_generate_png ( double **, const int, const char * );

#endif // _CIFAR_H_
