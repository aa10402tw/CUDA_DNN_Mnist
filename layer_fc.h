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
    // Define layer structure and activiation function
    int in_features;
    int out_features;
    ActivationFunction* activation;

    // Layer weight and bias
    GpuMat *weight;
    GpuMat *bias;

    // Record input, output before activation and output for compute gradient 
    GpuMat *input;
    GpuMat *preAct;
    GpuMat *output;

    // Gradient for weight and bias
    GpuMat *delta;
    GpuMat *last_grad_w;
    GpuMat *last_grad_b;
    GpuMat *grad_weight;
    GpuMat *grad_bias;

    // Record number of sample passed (to average gradient)
    unsigned int n_samples;

    // Constructor and Destructor
    Layer_fc(int in_features, int out_features);
    Layer_fc(int in_features, int out_features, ActivationFunction* activation);
    ~Layer_fc();

    // Forward-pass, Back-propgation, and Update-weight functions 
    GpuMat* forwardPass(const GpuMat *input);
    void backProp(const GpuMat* dLoss); // For last layer
    void backProp(Layer_fc *next_layer);
    void update(float lr);
    void reset_grad();
};

#endif
