#include "cifar.h"

#define CIFAR_DEBUG 0
#define CIFAR_DEBUG2 0
#define CIFAR_DEBUG3 0

static void cifar_read_training_image ( double ***training_image, int **training_label, int data_size )
{
  // malloc
  *training_image = (double **)malloc(data_size * sizeof(double *));
  for (int i = 0; i < data_size; i++) {
    (*training_image)[i] = (double *)malloc(CIFAR_IMAGE_PIXEL * sizeof(double));
  }
  *training_label = (int *)malloc(data_size * sizeof(int));

  // for grayscaling
  double *data = (double *)malloc(CIFAR_IMAGE_SIZE * sizeof(double));

  // file open
  FILE *file = fopen ( CIFAR_TRAINING_BINARY_FILE1, "rb" );
  if ( file == NULL ) {
    fprintf ( stderr, "err fopen %s\n", CIFAR_TRAINING_BINARY_FILE1 );
    exit ( 1 );
  }

  // read file
  for (int i = 0; i < 10000; i++) {
    unsigned char buf;
    // label
    if (! fread(&buf, sizeof(unsigned char), 1, file)) exit(1);
    (*training_label)[i] = buf;
    // image
    for (int j = 0; j < CIFAR_IMAGE_SIZE; j++) {
      if (! fread(&buf, sizeof(unsigned char), 1, file)) exit(1);
      data[j] = buf;
    }
    // grayscaling
    for (int j = 0; j < CIFAR_IMAGE_PIXEL; j++) {
      double sum = data[j] + data[j + CIFAR_IMAGE_PIXEL] + data[j + 2 * CIFAR_IMAGE_PIXEL];
      (*training_image)[i][j] = sum / 255.0 / 3;
    }
  }
  fclose(file);

  // file open
  file = fopen ( CIFAR_TRAINING_BINARY_FILE2, "rb" );
  if ( file == NULL ) {
    fprintf ( stderr, "err fopen %s\n", CIFAR_TRAINING_BINARY_FILE2 );
    exit ( 1 );
  }

  // read file
  for (int i = 10000; i < 20000; i++) {
    unsigned char buf;
    // label
    if (! fread(&buf, sizeof(unsigned char), 1, file)) exit(1);
    (*training_label)[i] = buf;
    // image
    for (int j = 0; j < CIFAR_IMAGE_SIZE; j++) {
      if (! fread(&buf, sizeof(unsigned char), 1, file)) exit(1);
      data[j] = buf;
    }
    // grayscaling
    for (int j = 0; j < CIFAR_IMAGE_PIXEL; j++) {
      double sum = data[j] + data[j + CIFAR_IMAGE_PIXEL] + data[j + 2 * CIFAR_IMAGE_PIXEL];
      (*training_image)[i][j] = sum / 255.0 / 3;
    }
  }
  fclose(file);

  // file open
  file = fopen ( CIFAR_TRAINING_BINARY_FILE3, "rb" );
  if ( file == NULL ) {
    fprintf ( stderr, "err fopen %s\n", CIFAR_TRAINING_BINARY_FILE3 );
    exit ( 1 );
  }

  // read file
  for (int i = 20000; i < 30000; i++) {
    unsigned char buf;
    // label
    if (! fread(&buf, sizeof(unsigned char), 1, file)) exit(1);
    (*training_label)[i] = buf;
    // image
    for (int j = 0; j < CIFAR_IMAGE_SIZE; j++) {
      if (! fread(&buf, sizeof(unsigned char), 1, file)) exit(1);
      data[j] = buf;
    }
    // grayscaling
    for (int j = 0; j < CIFAR_IMAGE_PIXEL; j++) {
      double sum = data[j] + data[j + CIFAR_IMAGE_PIXEL] + data[j + 2 * CIFAR_IMAGE_PIXEL];
      (*training_image)[i][j] = sum / 255.0 / 3;
    }
  }
  fclose(file);

  // file open
  file = fopen ( CIFAR_TRAINING_BINARY_FILE4, "rb" );
  if ( file == NULL ) {
    fprintf ( stderr, "err fopen %s\n", CIFAR_TRAINING_BINARY_FILE4 );
    exit ( 1 );
  }

  // read file
  for (int i = 30000; i < 40000; i++) {
    unsigned char buf;
    // label
    if (! fread(&buf, sizeof(unsigned char), 1, file)) exit(1);
    (*training_label)[i] = buf;
    // image
    for (int j = 0; j < CIFAR_IMAGE_SIZE; j++) {
      if (! fread(&buf, sizeof(unsigned char), 1, file)) exit(1);
      data[j] = buf;
    }
    // grayscaling
    for (int j = 0; j < CIFAR_IMAGE_PIXEL; j++) {
      double sum = data[j] + data[j + CIFAR_IMAGE_PIXEL] + data[j + 2 * CIFAR_IMAGE_PIXEL];
      (*training_image)[i][j] = sum / 255.0 / 3;
    }
  }
  fclose(file);

  // file open
  file = fopen ( CIFAR_TRAINING_BINARY_FILE5, "rb" );
  if ( file == NULL ) {
    fprintf ( stderr, "err fopen %s\n", CIFAR_TRAINING_BINARY_FILE5 );
    exit ( 1 );
  }

  // read file
  for (int i = 40000; i < 50000; i++) {
    unsigned char buf;
    // label
    if (! fread(&buf, sizeof(unsigned char), 1, file)) exit(1);
    (*training_label)[i] = buf;
    // image
    for (int j = 0; j < CIFAR_IMAGE_SIZE; j++) {
      if (! fread(&buf, sizeof(unsigned char), 1, file)) exit(1);
      data[j] = buf;
    }
    // grayscaling
    for (int j = 0; j < CIFAR_IMAGE_PIXEL; j++) {
      double sum = data[j] + data[j + CIFAR_IMAGE_PIXEL] + data[j + 2 * CIFAR_IMAGE_PIXEL];
      (*training_image)[i][j] = sum / 255.0 / 3;
    }
  }
  fclose(file);
}

static void cifar_read_test_image ( double ***test_image, int **test_label, int data_size )
{
  // malloc
  *test_image = (double **)malloc(data_size * sizeof(double *));
  for (int i = 0; i < data_size; i++) {
    (*test_image)[i] = (double *)malloc(CIFAR_IMAGE_PIXEL * sizeof(double));
  }
  *test_label = (int *)malloc(data_size * sizeof(int));

  // for grayscaling
  double *data = (double *)malloc(CIFAR_IMAGE_SIZE * sizeof(double));

  // file open
  FILE *file = fopen ( CIFAR_TEST_BINARY_FILE, "rb" );
  if ( file == NULL ) {
    fprintf ( stderr, "err fopen %s\n", CIFAR_TEST_BINARY_FILE );
    exit ( 1 );
  }

  // read file
  for (int i = 0; i < data_size; i++) {
    unsigned char buf;
    // label
    if (! fread(&buf, sizeof(unsigned char), 1, file)) exit(1);
    (*test_label)[i] = buf;
    // image
    for (int j = 0; j < CIFAR_IMAGE_SIZE; j++) {
      if (! fread(&buf, sizeof(unsigned char), 1, file)) exit(1);
      data[j] = buf;
    }
    // grayscaling
    for (int j = 0; j < CIFAR_IMAGE_PIXEL; j++) {
      double sum = data[j] + data[j + CIFAR_IMAGE_PIXEL] + data[j + 2 * CIFAR_IMAGE_PIXEL];
      (*test_image)[i][j] = sum / 255.0 / 3;
    }
  }
  fclose(file);
}

static void cifar_read_text_file ( const char *filename, char ***label, int data_size )
{

  FILE *file = fopen ( filename, "r" );

  if ( file == NULL ) {
    fprintf ( stderr, "err fopen %s\n", filename );
    exit ( 1 );
  }

  // malloc
  *label = (char **)malloc(data_size * sizeof(char *));
  for (int i = 0; i < data_size; i++) {
    (*label)[i] = (char *)malloc(64 * sizeof(char));
  }

  // read file
  for (int i = 0; i < data_size; i++) {
    char buf[64];
    if (! fgets(buf, 64, file)) exit(1);
    strcpy((*label)[i], buf);
  }

  fclose(file);
}

void cifar_initialize ( double ***training_image, int **training_label, double ***test_image, int **test_label, char ***label )
{
  cifar_read_training_image ( training_image, training_label, CIFAR_TRAINING_DATA_SIZE );
  cifar_read_test_image ( test_image, test_label, CIFAR_TEST_DATA_SIZE );
  cifar_read_text_file ( CIFAR_LABEL_FILE, label, CIFAR_LABEL_SIZE );
}

void cifar_finalize ( double **training_image, int *training_label, double **test_image, int *test_label, char **label )
{
  for (int i = 0; i < CIFAR_TRAINING_DATA_SIZE; i++) {
    free(training_image[i]);
  }
  free(training_image);
  free(training_label);
 
  for (int i = 0; i < CIFAR_TEST_DATA_SIZE; i++) {
    free(test_image[i]);
  }
  free(test_image);
  free(test_label);

  for (int i = 0; i < CIFAR_LABEL_SIZE; i++) {
    free(label[i]);
  }
  free(label);
}

void cifar_generate_png ( double **data, const int n, const char *filename )
{
  gdImagePtr im = gdImageCreate(CIFAR_IMAGE_ROW_SIZE, CIFAR_IMAGE_COL_SIZE);

  const int n_grayscale = 256;
  int gray[n_grayscale];
  for (int i = 0; i < n_grayscale; i++) {
    gray[i] = gdImageColorAllocate(im, i, i, i);
  }

  for (int i = 0; i < CIFAR_IMAGE_ROW_SIZE; i++) {
    for (int j = 0; j < CIFAR_IMAGE_COL_SIZE; j++) {
      int index = (int)((n_grayscale - 1)
			* data[n][j + CIFAR_IMAGE_COL_SIZE * i]);
      if ( CIFAR_DEBUG && CIFAR_DEBUG3 ) { printf ( "%d ", index ); }
      gdImageSetPixel(im, j, i, gray[index]);
    }
  }
  
  {
    FILE *file = fopen(filename, "wb");
    gdImagePng(im, file);
    fclose(file);
  }

  gdImageDestroy(im);
  
  return;
}

int local_main(void)
{
  double **training_image, **test_image;
  int *training_label, *test_label;
  char **label;

  cifar_initialize(&training_image, &training_label, &test_image, &test_label, &label);
  
  {  // Demo: generate 60 png files while printing corresponding labels
    for ( int i = 0; i < CIFAR_TRAINING_DATA_SIZE; i += 1000 ) {
      char fn [ 1024 ];
      sprintf ( fn, "./png/cifar_gray_%d.png", i );
      cifar_generate_png ( training_image, i, fn );
      printf ( "%s", label[training_label [ i ]] );
    }
  }
  
  cifar_finalize(training_image, training_label, test_image, test_label, label);
  
  return 0;
}

#if 0
int main(void)
{
  return local_main();
}
#endif
