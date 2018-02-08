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

void copyConnection(Network *network_src, int layer_id_src, Network *network_dst, int layer_id_dst)
{
  Connection *c_src, *c_dst;
  c_src = &network_src->connection[layer_id_src];
  c_dst = &network_dst->connection[layer_id_dst];
  int n_pre = c_src->n_pre;
  int n_post = c_src->n_post;
	
  for (int i = 0; i < n_post; i++) {
    for (int j = 0; j < n_pre; j++) {
      c_dst->w[j + n_pre * i] = c_src->w[j + n_pre * i];
    }
  }
}

void copyConnectionWithTranspose(Network *network_src, int layer_id_src, Network *network_dst, int layer_id_dst)
{
  Connection *c_src, *c_dst;
  c_src = &network_src->connection[layer_id_src];
  c_dst = &network_dst->connection[layer_id_dst];
  int n_pre_src = c_src->n_pre;
  int n_post_src = c_src->n_post;
  int n_pre_dst = c_dst->n_pre;
  int n_post_dst = c_dst->n_post;

  for (int i = 0; i < n_post_src; i++) {
    for (int j = 0; j < n_post_dst; j++) {
      c_dst->w[i + n_pre_dst * j] = c_src->w[j + n_pre_src * i];
    }
  }
}

double updateByBackPropagationPartial(Network *network, Neuron z[])
{
  const double Eta = 0.1;
	
  double error = 0.;
  {
    Layer *l = &network->layer[network->n - 1];
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
  l = l - 1;
  post_layer = &network->layer[l + 1];
  current_layer = &network->layer[l];
  pre_layer = &network->layer[l - 1];
  post_c = &network->connection[l];
  pre_c = &network->connection[l - 1];
  for (int i = 0; i < pre_c->n_post; i++) {
    delta = 0.;
    for (int k = 0; k < post_c->n_post; k++) {
      o = post_layer->z[k];
      delta += post_layer->delta[k] * o * (1 - o) * post_c->w[i + post_c->n_pre * k];
    }
    current_layer->delta[i] = delta;
    o = current_layer->z[i];
    d = delta * o * (1 - o);
    for (int j = 0; j < pre_c->n_pre; j++) {
      pre_c->dw[j + pre_c->n_pre * i] += Eta * d * (pre_layer->z[j]);
    }
  }

  return error;
}

void connection_generate_png(double *data, const char *filename)
{
  gdImagePtr im = gdImageCreate(MNIST_IMAGE_ROW_SIZE, MNIST_IMAGE_COL_SIZE);
	
  const int n_grayscale = 256;
  int gray[n_grayscale];
  for (int i = 0; i < n_grayscale; i++) {
    gray[i] = gdImageColorAllocate(im, i, i, i);
  }

  for (int i = 0; i < MNIST_IMAGE_ROW_SIZE; i++) {
    for (int j = 0; j < MNIST_IMAGE_COL_SIZE; j++) {
      int index = (int)((n_grayscale - 1) * data[j + MNIST_IMAGE_COL_SIZE * i]);
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
  fprintf(stderr, "Network\n");
  fprintf(stderr, "Number of Layers: %d\n", n->n);
  fprintf(stderr, "  Input Layer\n");
  for (i = 0; i < n->n - 1; i++) {
    fprintf(stderr, "  Layer %3d\n", i);
    fprintf(stderr, "  Number of Neurons: %d\n", l[i].n);
    for (j = 0; j < l[i].n; j++) {
      fprintf(stderr, "    Neuron %3d: %10.7f\n", j, l[i].z[j]);
      for (k = 0; k < c[i].n_post; k++) {
	fprintf(stderr, "      (%3d->%3d): %10.7f\n", j, k, c[i].w[j + c[i].n_pre * k]);
      }
    }
    fprintf(stderr, "    Bias: %10.7f\n", l[i].z[j]);
    for (k = 0; k < c[i].n_post; k++) {
      fprintf(stderr, "      (Bias->%3d): %10.7f\n", k, c[i].w[j + c[i].n_pre * k]);
    }
  }
  fprintf(stderr, "  Output Layer\n");
  fprintf(stderr, "  Layer %3d\n", i);
  fprintf(stderr, "  Number of Neurons: %d\n", l[i].n);
  for (j = 0; j < l[i].n; j++) {
    fprintf(stderr, "    Neuron %3d: %10.7f\n", j, l[i].z[j]);
  }
}

int main(void)
{
  double **training_image, **test_image;
  int *training_label, *test_label;
  mnist_initialize(&training_image, &training_label, &test_image, &test_label);

  sfmt_t rng;
  sfmt_init_gen_rand ( &rng, getpid ( ) );

  /********************************
   *      Create Network 1        *
   ********************************/
  Network network1;
  createNetwork ( &network1, 3, rng );
  createLayer ( &network1, 0, MNIST_IMAGE_SIZE );
  createLayer ( &network1, 1, 128 );
  createLayer ( &network1, 2, MNIST_IMAGE_SIZE );
  createConnection ( &network1, 0, uniform_random );
  createConnection ( &network1, 1, uniform_random );
  copyConnectionWithTranspose(&network1, 0, &network1, 1); // tied weight

   // Training 1
  for (int i = 0; i < MNIST_TRAINING_DATA_SIZE; i++) {
    initializeDW(&network1);
    double error = 0.;
    // エポックの先頭で毎回転置をとってコピー
    copyConnectionWithTranspose(&network1, 0, &network1, 1);
    for (int j = 0; j < 10; j++) {
      int k = (int)(MNIST_TRAINING_DATA_SIZE * sfmt_genrand_real2(&rng));
      //int k = i; // use all training data one by one sequentially
      setInput(&network1, training_image[k]);
      forwardPropagation(&network1, sigmoid);
      // 入力層の活動を教師信号としてupdateByBackPropagationPartial
      error += updateByBackPropagationPartial(&network1, network1.layer[0].z);
    }
    printf("%d %f\n", i, error);
    updateW(&network1);
  }


  /********************************
   *      Create Network 2        *
   ********************************/
  Network network2;
  sfmt_init_gen_rand ( &rng, getpid ( ) + 1 );
  createNetwork ( &network2, 4, rng );
  createLayer ( &network2, 0, MNIST_IMAGE_SIZE );
  createLayer ( &network2, 1, 128 );
  createLayer ( &network2, 2, 64 );
  createLayer ( &network2, 3, 128 );
  createConnection ( &network2, 0, uniform_random );
  createConnection ( &network2, 1, uniform_random );
  createConnection ( &network2, 2, uniform_random );
  copyConnection(&network1, 0, &network2, 0); // copy weight of network1
  copyConnectionWithTranspose(&network2, 1, &network2, 2); // tied weight


  // Delete Network1
  deleteConnection ( &network1, 1 );
  deleteConnection ( &network1, 0 );
  deleteLayer ( &network1, 2 );
  deleteLayer ( &network1, 1 );
  deleteLayer ( &network1, 0 );
  deleteNetwork ( &network1 );


  // Training 2
  for (int i = 0; i < MNIST_TRAINING_DATA_SIZE; i++) {
    initializeDW(&network2);
    double error = 0.;
    // エポックの先頭で毎回転置をとってコピー
    copyConnectionWithTranspose(&network2, 1, &network2, 2);
    for (int j = 0; j < 10; j++) {
      int k = (int)(MNIST_TRAINING_DATA_SIZE * sfmt_genrand_real2(&rng));
      //int k = i; // use all training data one by one sequentially
      setInput(&network2, training_image[k]);
      forwardPropagation(&network2, sigmoid);
      // 入力層の活動を教師信号としてupdateByBackPropagationPartial
      error += updateByBackPropagationPartial(&network2, network2.layer[1].z);
    }
    printf("%d %f\n", i, error);
    updateW(&network2);
  }


  /********************************
   *      Create Network 3        *
   ********************************/
  Network network3;
  sfmt_init_gen_rand ( &rng, getpid ( ) + 2 );
  createNetwork ( &network3, 5, rng );
  createLayer ( &network3, 0, MNIST_IMAGE_SIZE );
  createLayer ( &network3, 1, 128 );
  createLayer ( &network3, 2, 64 );
  createLayer ( &network3, 3, 32 );
  createLayer ( &network3, 4, 64 );
  createConnection ( &network3, 0, uniform_random );
  createConnection ( &network3, 1, uniform_random );
  createConnection ( &network3, 2, uniform_random );
  createConnection ( &network3, 3, uniform_random );
  copyConnection(&network2, 0, &network3, 0); // copy weight of network2
  copyConnection(&network2, 1, &network3, 1); // copy weight of network2
  copyConnectionWithTranspose(&network3, 2, &network3, 3); // tied weight


  // Delete Network2
  deleteConnection ( &network2, 2 );
  deleteConnection ( &network2, 1 );
  deleteConnection ( &network2, 0 );
  deleteLayer ( &network2, 3 );
  deleteLayer ( &network2, 2 );
  deleteLayer ( &network2, 1 );
  deleteLayer ( &network2, 0 );
  deleteNetwork ( &network2 );


  // Training 3
  for (int i = 0; i < MNIST_TRAINING_DATA_SIZE; i++) {
    initializeDW(&network3);
    double error = 0.;
    // エポックの先頭で毎回転置をとってコピー
    copyConnectionWithTranspose(&network3, 2, &network3, 3);
    for (int j = 0; j < 10; j++) {
      int k = (int)(MNIST_TRAINING_DATA_SIZE * sfmt_genrand_real2(&rng));
      //int k = i; // use all training data one by one sequentially
      setInput(&network3, training_image[k]);
      forwardPropagation(&network3, sigmoid);
      // 入力層の活動を教師信号としてupdateByBackPropagationPartial
      error += updateByBackPropagationPartial(&network3, network3.layer[2].z);
    }
    printf("%d %f\n", i, error);
    updateW(&network3);
  }


  /********************************
   *      Create Network          *
   ********************************/
  Network network;
  sfmt_init_gen_rand ( &rng, getpid ( ) + 3 );
  createNetwork ( &network, 5, rng );
  createLayer ( &network, 0, MNIST_IMAGE_SIZE );
  createLayer ( &network, 1, 128 );
  createLayer ( &network, 2, 64 );
  createLayer ( &network, 3, 32 );
  createLayer ( &network, 4, MNIST_LABEL_SIZE );
  createConnection ( &network, 0, uniform_random );
  createConnection ( &network, 1, uniform_random );
  createConnection ( &network, 2, uniform_random );
  createConnection ( &network, 3, uniform_random );
  copyConnection(&network3, 0, &network, 0); // copy weight of network2
  copyConnection(&network3, 1, &network, 1); // copy weight of network2
  copyConnection(&network3, 2, &network, 2); // copy weight of network2


  // Delete Network3
  deleteConnection ( &network3, 3 );
  deleteConnection ( &network3, 2 );
  deleteConnection ( &network3, 1 );
  deleteConnection ( &network3, 0 );
  deleteLayer ( &network3, 4 );
  deleteLayer ( &network3, 3 );
  deleteLayer ( &network3, 2 );
  deleteLayer ( &network3, 1 );
  deleteLayer ( &network3, 0 );
  deleteNetwork ( &network3 );


  // Last Training 
  for (int i = 0; i < MNIST_TRAINING_DATA_SIZE; i++) {
    initializeDW(&network);
    double error = 0.;
    for (int j = 0; j < 10; j++) {
      int k = (int)(MNIST_TRAINING_DATA_SIZE * sfmt_genrand_real2(&rng));
      //int k = i; // use all training data one by one sequentially
      setInput(&network, training_image[k]);
      forwardPropagation(&network, sigmoid);
      double z[MNIST_LABEL_SIZE] = {0.};
      z[training_label[k]] = 1.;
      error += updateByBackPropagation(&network, z);
    }
    printf("%d %f\n", i, error);
    updateW(&network);
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


  // Dlete Network
  deleteConnection ( &network, 3 );
  deleteConnection ( &network, 2 );
  deleteConnection ( &network, 1 );
  deleteConnection ( &network, 0 );
  deleteLayer ( &network, 4 );
  deleteLayer ( &network, 3 );
  deleteLayer ( &network, 2 );
  deleteLayer ( &network, 1 );
  deleteLayer ( &network, 0 );
  deleteNetwork ( &network );

  mnist_finalize(training_image, training_label, test_image, test_label);

  return 0;
}
