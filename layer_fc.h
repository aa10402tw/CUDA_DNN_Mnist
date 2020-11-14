#include <cstdlib>
#include <vector>
#include <memory>
#include <cublas_v2.h>
#include <cuda.h>

#include "mat.h"
#include "mat_math.h"
#include "act.h"

#ifndef LAYER_FC_H
#define LAYER_FC_H

class Layer_fc;

class Layer_fc{
public:
    int in_features;
    int out_features;
    ActivationFunction* activation;

    unsigned int n_samples;
    gpuMat *bias;
    gpuMat *weight;

    gpuMat *input;
    gpuMat *preAct;
    gpuMat *output;

    gpuMat *delta;
    gpuMat *grad_bias;
    gpuMat *grad_weight;

    // Momentum
    gpuMat *last_grad_b;
    gpuMat *last_grad_w;

    Layer_fc(int in_features, int out_features);
    Layer_fc(int in_features, int out_features, ActivationFunction* activation);
    ~Layer_fc();

    gpuMat* forwardPass(const gpuMat *input);
    void backProp(const gpuMat* delta_out);
    void backProp(Layer_fc *next_layer);
    void update(float lr);
};

#endif
