#include "layer_fc.h"
#include "mat.h"
#include "mat_math.h"
#include "act.h"

#include <time.h>
#include <vector>
#include <iostream>

#define momentum 0.9

// Constructor
Layer_fc::Layer_fc(int in_features, int out_features) {
    this->in_features = in_features;
    this->out_features = out_features;
    this->activation = new Act_NONE();

    n_samples = 0;

    bias = new GpuMat(out_features, 1, 1);
    weight = new GpuMat(out_features, in_features, 1);

    input = new GpuMat();
    preAct = new GpuMat();
    output = new GpuMat();

    delta = new GpuMat(out_features, 1, 1);
    grad_bias = new GpuMat(out_features, 1, 1);
    grad_weight = new GpuMat(out_features, in_features, 1);
    last_grad_b = new GpuMat(out_features, 1, 1);
    last_grad_w = new GpuMat(out_features, in_features, 1);

    // Random Init weight and bias 
    weight->randn();
}

Layer_fc::Layer_fc(int in_features, int out_features, ActivationFunction* activation) {
    this->in_features = in_features;
    this->out_features = out_features;
    this->activation = activation;

    n_samples = 0;

    bias = new GpuMat(out_features, 1, 1);
    weight = new GpuMat(out_features, in_features, 1);

    input = new GpuMat();
    preAct = new GpuMat();
    output = new GpuMat();

    delta = new GpuMat(out_features, 1, 1);
    grad_bias = new GpuMat(out_features, 1, 1);
    grad_weight = new GpuMat(out_features, in_features, 1);
    last_grad_b = new GpuMat(out_features, 1, 1);
    last_grad_w = new GpuMat(out_features, in_features, 1);

    // Random Init weight and bias 
    weight->randn();
}

// Destructor
Layer_fc::~Layer_fc() {
    weight->release();
    bias->release();
    
    input->release();
    preAct->release();
    output->release();

    delta->release();
    last_grad_w->release();
    last_grad_b->release();
    grad_weight->release();
    grad_bias->release();
}

// Forward Pass function
GpuMat* Layer_fc::forwardPass(const GpuMat *_input) {
    if (_input->Data == nullptr) {
        std::cout << "Invalid Input (NULL) for forward pass";
        exit(0);
    }
    
    // Release last round data
    input->release();
    preAct->release();
    output->release();

    input = GpuMat::copy(_input);
    GpuMat *Wx = MatMul(weight, input);
    preAct = MatAdd(Wx, bias);
    output = activation->forwardPass(preAct);

    // Release tmp data
    Wx->release();

    return GpuMat::copy(output);
}

// Back propgation function for the last layer
void Layer_fc::backProp(const GpuMat* dLoss) {
    if (dLoss->Data == nullptr) {
        std::cout << "Invalid dLoss (NULL) for back-propgation";
        exit(0);
    }
    // Release last round data 
    delta->release();

    // Compute delta function [delta = dLoss x dF]
    GpuMat* dPreAct = activation->derivative(preAct);
    delta = MatEleMul(dLoss, dPreAct);

    // Compute gradient for weight and bias [grad_w = delta * input^T], [grad_b = delta]
    GpuMat* input_t = Transpose(input);
    GpuMat* new_grad_w = MatMul(delta, input_t);
    GpuMat* new_grad_b = GpuMat::copy(delta);

    if (n_samples == 0) {
        // If is the first sample, simply assign new gradient
        grad_weight->release();
        grad_bias->release();
        grad_weight = GpuMat::copy(new_grad_w);
        grad_bias = GpuMat::copy(new_grad_b);
    }
    else{
        // Else accumulate the gradient
        GpuMat* acc_grad_w = MatAdd(grad_weight, new_grad_w);
        GpuMat* acc_grad_b = MatAdd(grad_bias, new_grad_b);
        cudaMemcpy(grad_weight->Data, acc_grad_w->Data, grad_weight->getSize() * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(grad_bias->Data, acc_grad_b->Data, grad_bias->getSize() * sizeof(float), cudaMemcpyDeviceToDevice);
        acc_grad_w->release();
        acc_grad_b->release();
    }

    // Add sample number
    n_samples += 1;

    // Release tmp data
    dPreAct->release();
    input_t->release();
    new_grad_w->release();
    new_grad_b->release();
}

// Back propgation function
void Layer_fc::backProp(Layer_fc *next_layer) {
    if (next_layer->delta == nullptr) {
        std::cout << "Invalid delta_next (NULL) for back-propgation";
        exit(0);
    }
    // Release last round data 
    delta->release();

    // Compute delta function
    GpuMat* w_next_t = Transpose(next_layer->weight);
    GpuMat* delta_next = GpuMat::copy(next_layer->delta);

    GpuMat* next_w_delta = MatMul(w_next_t, delta_next);
    GpuMat* dPreAct = activation->derivative(preAct);
    delta = MatEleMul(next_w_delta, dPreAct);

    // Compute gradient for weight and bias [grad_w = delta * input^T], [grad_b = delta]
    GpuMat* input_t = Transpose(input);
    GpuMat* new_grad_w = MatMul(delta, input_t);
    GpuMat* new_grad_b = GpuMat::copy(delta);

    if (n_samples == 0) {
        // If is the first sample, simply assign new gradient
        grad_weight->release();
        grad_bias->release();
        grad_weight = GpuMat::copy(new_grad_w);
        grad_bias = GpuMat::copy(new_grad_b);
    }
    else{
        // Else accumulate the gradient
        GpuMat* acc_grad_w = MatAdd(grad_weight, new_grad_w);
        GpuMat* acc_grad_b = MatAdd(grad_bias, new_grad_b);
        cudaMemcpy(grad_weight->Data, acc_grad_w->Data, grad_weight->getSize() * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(grad_bias->Data, acc_grad_b->Data, grad_bias->getSize() * sizeof(float), cudaMemcpyDeviceToDevice);
        acc_grad_w->release();
        acc_grad_b->release();
    }

    // Add sample number
    n_samples += 1;

    // Release tmp data
    dPreAct->release();
    w_next_t->release();
    delta_next->release();
    next_w_delta->release();
    new_grad_w->release();
    input_t->release();
    new_grad_b->release();
}

// Update the weight of layer
void Layer_fc::update(float lr) {
    if (n_samples == 0  || grad_weight->Data == nullptr || grad_bias->Data == nullptr) {
        std::cout << "Invalid gradient for update";
        exit(0);
    }

    // Update weight [new_w = w - lr*grad_w] using momentum [grad_w = a*last_grad_w+(1-a)*new_grad_w] 
    GpuMat* a_last_grad_w = MatEleMul(last_grad_w, momentum);
    GpuMat* new_grad_w = MatEleDiv(grad_weight, (float)n_samples); // average gradient
    GpuMat* b_new_grad_w = MatEleMul(new_grad_w, (1-momentum)); 
    GpuMat* grad_w = MatAdd(a_last_grad_w, b_new_grad_w);
    GpuMat* step_w = MatEleMul(grad_w, lr);
    GpuMat* new_w = MatSub(weight, step_w);
    cudaMemcpy(weight->Data, new_w->Data, weight->getSize() * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(last_grad_w->Data, grad_w->Data, last_grad_w->getSize() * sizeof(float), cudaMemcpyDeviceToDevice);

    // Update bias [new_b = b - lr*grad_b] using momentum [grad_b = a*last_grad_b+(1-a)*new_grad_b] 
    GpuMat* a_last_grad_b = MatEleMul(last_grad_b, momentum);
    GpuMat* new_grad_b = MatEleDiv(grad_bias, (float)n_samples); // average gradient
    GpuMat* b_new_grad_b = MatEleMul(new_grad_b, (1-momentum)); 
    GpuMat* grad_b = MatAdd(a_last_grad_b, b_new_grad_b);
    GpuMat* step_b = MatEleMul(grad_b, lr);
    GpuMat* new_b = MatSub(bias, step_b);
    cudaMemcpy(bias->Data, new_b->Data, bias->getSize() * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(last_grad_b->Data, grad_b->Data, last_grad_b->getSize() * sizeof(float), cudaMemcpyDeviceToDevice);

    // Release tmp data for compute weight
    a_last_grad_w->release();
    new_grad_w->release();
    b_new_grad_w->release();
    grad_w->release();
    step_w->release();
    new_w->release();

    // Release tmp data for compute bias
    a_last_grad_b->release();
    new_grad_b->release();
    b_new_grad_b->release();
    grad_b->release();
    step_b->release();
    new_b->release();

    // Release iteration data
    reset_grad();
}

// Reset the gradient and clear iteration variable
void Layer_fc::reset_grad() {
    // Release iteration data
    input->release();
    preAct->release();
    output->release();
    grad_bias->release();
    grad_weight->release();
    delta->release();
    n_samples=0;
}







