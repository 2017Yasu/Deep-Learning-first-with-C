/*
  make mnist_test
  source ~ya000836/dl/environment.csh
  ./mnist_test 
  gcc -O3 -std=c99 -Wall -I ~ya000836/usr/include/ -I SFMT-src-1.5.1 -D SFMT_MEXP=19937 -o rec_mnist rec_mnist.c mnist.c SFMT-src-1.5.1/SFMT.c -lm
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <SFMT.h>
#include "mnist.h"

#define MNIST_DEBUG 0
#define MNIST_DEBUG2 0
#define MNIST_DEBUG3 0

extern void sfmt_init_gen_rand(sfmt_t * sfmt, uint32_t seed);
extern double sfmt_genrand_real2(sfmt_t * sfmt);

typedef double Neuron, Delta, Weight;
typedef struct {
  Weight *w;
  Weight *dw;
  int n_pre;
  int n_post;
} Connection;
typedef struct {
  Neuron *z;
  Delta *delta;
  int n;
} Layer;
typedef struct {
  Layer *layer;
  Connection *connection;
  sfmt_t rng;
  int n;
} Network;

double all_to_all(Network *n, const int i, const int j) { return 1.; }
double uniform_random(Network *n, const int i, const int j)
{
  return 1. - 2. * sfmt_genrand_real2(&n->rng);
}
double sparse_random(Network *n, const int i, const int j)
{
  return (sfmt_genrand_real2(&n->rng) < 0.5) ? uniform_random(n, i, j) : 0.;
}
double sigmoid(double x) { return 1. / (1. + exp(-x)); }

void createNetwork(Network *network, const int number_of_layers, const sfmt_t rng)
{
  network->layer = (Layer *)malloc(number_of_layers * sizeof(Layer));
  network->connection = (Connection *)malloc(number_of_layers * sizeof(Connection));
  network->n = number_of_layers;
  network->rng = rng;
}

void deleteNetwork(Network *network)
{
  free(network->layer);
  free(network->connection);
}

void createLayer(Network *network, const int layer_id, const int number_of_neurons)
{
  Layer *layer = &network->layer[layer_id];

  layer->n = number_of_neurons;

  int bias = (layer_id < network->n - 1) ? 1 : 0; // 出力層以外はバイアスを用意

  layer->z = (Neuron *)malloc((number_of_neurons + bias) * sizeof(Neuron));
  for (int i = 0; i < layer->n; i++) { layer->z[i] = 0.; } // 初期化
  if (bias) { layer->z[layer->n] = +1.; } // バイアス初期化

  // Deltaを追加
  layer->delta = (Delta *)malloc((number_of_neurons + bias) * sizeof(Delta));
  for (int i = 0; i < layer->n; i++) { layer->delta[i] = 0.; }
  if (bias) { layer->delta[layer->n] = 0.; } // バイアス初期化

}

void deleteLayer(Network *network, const int layer_id)
{
  Layer *layer = &network->layer[layer_id];
  free(layer->z);
  free(layer->delta);
}

void createConnection(Network *network, const int layer_id, double (*func)(Network *, const int, const int))
{
  Connection *connection = &network->connection[layer_id];

  const int n_pre = network->layer[layer_id].n + 1; // +1 for bias
  const int n_post = (layer_id == network->n - 1 ) ? 1 : network->layer[layer_id + 1].n;

  connection->w = (Weight *)malloc(n_pre * n_post * sizeof(Weight));
  for (int i = 0; i < n_post; i++) {
    for (int j = 0; j < n_pre; j++) {
      connection->w[j + n_pre * i] = func(network, i, j);
    }
  }

  connection->dw = (Weight *)malloc(n_pre * n_post * sizeof(Weight));
  for (int i = 0; i < n_post; i++) {
    for (int j = 0; j < n_pre; j++) {
      connection->dw[j + n_pre * i] = 0.;
    }
  }

  connection->n_pre = n_pre;
  connection->n_post = n_post;
}

void deleteConnection(Network *network, const int layer_id)
{
  Connection *connection = &network->connection[layer_id];
  free(connection->w);
  free(connection->dw);
}

void setInput(Network *network, Neuron x[])
{
  Layer *input_layer = &network->layer[0];
  for (int i = 0; i < input_layer->n; i++) {
    input_layer->z[i] = x[i];
  }
}

void forwardPropagation(Network *network, double (*activation)(double))
{
  for (int i = 0; i < network->n - 1; i++) {
    Layer *l_pre = &network->layer[i];
    Layer *l_post = &network->layer[i + 1];
    Connection *c = &network->connection[i];
    for (int j = 0; j < c->n_post; j++) {
      Neuron u = 0.;
      for (int k = 0; k < c->n_pre; k++) {
	u += (c->w[k + c->n_pre * j]) * (l_pre->z[k]);
      }
      l_post->z[j] = activation(u);
    }
  }
}

double updateByBackPropagation(Network *network, Neuron z[])
{
  const double Eta = 0.1;
	
  double error = 0.;
  {
    Layer *l = &network->layer[network->n -1];
    for (int j = 0; j < l->n; j++) {
      error += 0.5 * ((l->z[j] - z[j]) * (l->z[j] - z[j]));
    }
  }

  /* calculate delta */
  int l = network->n - 1;
  Layer *post_layer, *current_layer, *pre_layer;
  Connection *post_c, *pre_c;
  double o, d, delta;

  /* current_layer = output layer */
  current_layer = &network->layer[l];
  pre_layer = &network->layer[l - 1];
  pre_c = &network->connection[l - 1];
  for (int i = 0; i < pre_c->n_post; i++) {
    o = current_layer->z[i];
    delta = z[i] - o;
    d = delta * o * (1 - o);
    current_layer->delta[i] = delta;
    for (int j = 0; j < pre_c->n_pre; j++) {
      pre_c->dw[j + pre_c->n_pre * i] += Eta * d * (pre_layer->z[j]);
    }
  }

  /* the other layer */
  for (l = l - 1; l > 0; l--) {
    post_layer = &network->layer[l + 1];
    current_layer = &network->layer[l];
    pre_layer = &network->layer[l - 1];
    post_c = &network->connection[l];
    pre_c = &network->connection[l - 1];
    for (int i = 0; i < pre_c->n_post; i++) {
      delta = 0.;
      for (int k = 0; k < post_c->n_post; k++) {
	o = post_layer->z[k];
	delta += post_layer->delta[k] * o *
	  (1 - o) * post_c->w[i + post_c->n_pre * k];
      }
      current_layer->delta[i] = delta;
      o = current_layer->z[i];
      d = delta * o * (1 - o);
      for (int j = 0; j < pre_c->n_pre; j++) {
	pre_c->dw[j + pre_c->n_pre * i] += Eta * d * (pre_layer->z[j]);
      }
    }
  }

  return error;
}

void initializeDW(Network *network)
{
  for (int layer_id = 0; layer_id < network->n - 1; layer_id++) {
    Connection *c = &network->connection[layer_id];
    for (int i = 0; i < c->n_post; i++) {
      for (int j = 0; j < c->n_pre; j++) {
	c->dw[j + c->n_pre * i] = 0.;
      }
    }
  }
}

void updateW(Network *network)
{
  for (int layer_id = 0; layer_id < network->n - 1; layer_id++) {
    Connection *c = &network->connection[layer_id];
    for (int i = 0; i < c->n_post; i++) {
      for (int j = 0; j < c->n_pre; j++) {
	c->w[j + c->n_pre * i] += c->dw[j + c->n_pre * i];
      }
    }
  }
}

void connection_generate_png(double *data, const char *filename)
{
  gdImagePtr im = gdImageCreate(MNIST_IMAGE_ROW_SIZE, MNIST_IMAGE_COL_SIZE);
	
  const int n_grayscale = 256;
  int gray[n_grayscale];
  for (int i = 0; i < n_grayscale; i++) {
    gray[i] = gdImageColorAllocate(im, i, i, i);
  }

  // for Normalization
  double max = -1000.0, min = 1000.0, range;
  for (int i = 0; i < MNIST_IMAGE_ROW_SIZE; i++) {
    for (int j = 0; j < MNIST_IMAGE_COL_SIZE; j++) {
      if (max < data[j + MNIST_IMAGE_COL_SIZE * i])
	max = data[j + MNIST_IMAGE_COL_SIZE * i];
      if (min > data[j + MNIST_IMAGE_COL_SIZE * i])
	min = data[j + MNIST_IMAGE_COL_SIZE * i];
    }
  }
  range = max - min;

  for (int i = 0; i < MNIST_IMAGE_ROW_SIZE; i++) {
    for (int j = 0; j < MNIST_IMAGE_COL_SIZE; j++) {
      // Normalization
      double x = (data[j + MNIST_IMAGE_COL_SIZE * i] - min) / range;
      int index = (int)((n_grayscale - 1) * x);
      if (MNIST_DEBUG && MNIST_DEBUG3) {
	printf("%d ", index);
      }
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

void dump(Network *network)
{
  int i, j, k;
  Network *n = network;
  Layer *l = n->layer;
  Connection *c = n->connection;
  printf("Network\n");
  printf("Number of Layers: %d\n", n->n);
  printf("  Input Layer\n");
  for (i = 0; i < n->n - 1; i++) {
    printf("  Layer %3d\n", i);
    printf("  Number of Neurons: %d\n", l[i].n);
    for (j = 0; j < l[i].n; j++) {
      printf("    Neuron %3d: %10.7f\n", j, l[i].z[j]);
      for (k = 0; k < c[i].n_post; k++) {
	printf("      (%3d->%3d): %10.7f\n", j, k, c[i].w[j + c[i].n_pre * k]);
      }
    }
    printf("    Bias: %10.7f\n", l[i].z[j]);
    for (k = 0; k < c[i].n_post; k++) {
      printf("      (Bias->%3d): %10.7f\n", k, c[i].w[j + c[i].n_pre * k]);
    }
  }
  printf("  Output Layer\n");
  printf("  Layer %3d\n", i);
  printf("  Number of Neurons: %d\n", l[i].n);
  for (j = 0; j < l[i].n; j++) {
    printf("    Neuron %3d: %10.7f\n", j, l[i].z[j]);
  }
}

int main(void)
{
  double **training_image, **test_image;
  int *training_label, *test_label;
  mnist_initialize(&training_image, &training_label, &test_image, &test_label);

  sfmt_t rng;
  sfmt_init_gen_rand ( &rng, getpid ( ) );

  int neuron_hid = 32; // neuron of hidden layer
  int num_bat = 5; // number of batch
  int epoch = MNIST_TRAINING_DATA_SIZE;
  fprintf(stderr, "input neuron of hidden layer: (default %d)", neuron_hid);
  scanf("%d", &neuron_hid);
  fprintf(stderr, "input number of batch: (default %d)", num_bat);
  scanf("%d", &num_bat);
  fprintf(stderr, "input epoch: (default %d)", epoch);
  scanf("%d", &epoch);

  Network network;
  createNetwork ( &network, 3, rng );
  createLayer ( &network, 0, MNIST_IMAGE_SIZE );
  createLayer ( &network, 1, neuron_hid );
  createLayer ( &network, 2, MNIST_LABEL_SIZE );
  createConnection ( &network, 0, sparse_random );
  createConnection ( &network, 1, uniform_random );

  fprintf(stderr, "\nneuron of hidden layer: %d\n", neuron_hid);
  fprintf(stderr, "number of batch: %d\n", num_bat);
  fprintf(stderr, "epoch: %d\n", epoch);
	
  // Training
  fprintf(stderr, "data size: %d\n\n", MNIST_TRAINING_DATA_SIZE);
  for (int i = 0; i < epoch; i++) {
    initializeDW(&network);
    double error = 0.;
    for (int j = 0; j < num_bat; j++) {
      int k = (int)(MNIST_TRAINING_DATA_SIZE * sfmt_genrand_real2(&rng));
      //int k = i; // use all training data one by one sequentially
      setInput(&network, training_image[k]);
      forwardPropagation(&network, sigmoid);
      double z[MNIST_LABEL_SIZE] = {0.};
      z[training_label[k]] = 1.;
      error += updateByBackPropagation(&network, z);
    }
    updateW(&network);
    //printf( "%d %f\n", i, error);
  }	

  // Evaluation
  Layer *output_layer = &network.layer[network.n - 1];
  const int n = output_layer->n;
  int correct = 0;
  for (int i = 0; i < MNIST_TEST_DATA_SIZE; i++) {
    setInput(&network, test_image[i]);
    forwardPropagation(&network, sigmoid);
    int maxj = 0;
    double maxz = 0;
    for (int j = 0; j < n; j++) {
      if (output_layer->z[j] > maxz) {
	maxz = output_layer->z[j];
	maxj = j;
      }
    }
    correct += (maxj == test_label[i]);
  }
  fprintf(stderr, "success rate = %f\n", (double)correct / MNIST_TEST_DATA_SIZE);

  // generate png file with connection
  Connection *c = &network.connection[0];
  for (int i = 0; i < neuron_hid; i++) {
    char fn[1024];
    sprintf(fn, "./con_png/neu_sigmoid_%d.png", i);
    connection_generate_png(&c->w[i * MNIST_IMAGE_SIZE], fn);
  }

  deleteConnection ( &network, 1 );
  deleteConnection ( &network, 0 );
  deleteLayer ( &network, 2 );
  deleteLayer ( &network, 1 );
  deleteLayer ( &network, 0 );
  deleteNetwork ( &network );

  mnist_finalize(training_image, training_label, test_image, test_label);

  return 0;
}
