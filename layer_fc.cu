#include "layer_fc.h"
#include "mat.h"
#include "mat_math.h"
#include "act.h"

#include <time.h>
#include <vector>
#include <iostream>

// Constructor
Layer_fc::Layer_fc(int in_features, int out_features) {
    this->in_features = in_features;
    this->out_features = out_features;
    this->activation = new Act_ReLU();

    n_samples = 0;

    bias = new gpuMat(out_features, 1, 1);
    weight = new gpuMat(out_features, in_features, 1);

    input = new gpuMat();
    preAct = new gpuMat();
    output = new gpuMat();

    delta = new gpuMat(out_features, 1, 1);
    grad_bias = new gpuMat(out_features, 1, 1);
    grad_weight = new gpuMat(out_features, in_features, 1);
    last_grad_b = new gpuMat(out_features, 1, 1);
    last_grad_w = new gpuMat(out_features, in_features, 1);

    /* Init weight and bias */
    bias->randn();
    weight->randn();
}

Layer_fc::Layer_fc(int in_features, int out_features, ActivationFunction* activation) {
    this->in_features = in_features;
    this->out_features = out_features;
    this->activation = activation;

    bias = new gpuMat(out_features, 1, 1);
    weight = new gpuMat(out_features, in_features, 1);

    input = new gpuMat();
    preAct = new gpuMat();
    output = new gpuMat();

    delta = new gpuMat(out_features, 1, 1);
    grad_bias = new gpuMat(out_features, 1, 1);
    grad_weight = new gpuMat(out_features, in_features, 1);
    last_grad_b = new gpuMat(out_features, 1, 1);
    last_grad_w = new gpuMat(out_features, in_features, 1);

    /* Init weight and bias */
    bias->randn();
    weight->randn();
}

// Destructor
Layer_fc::~Layer_fc() {
    bias->release();
    weight->release();

    output->release();
    grad_bias->release();
    grad_weight->release();
}

gpuMat* Layer_fc::forwardPass(const gpuMat *_input) {
    n_samples += 1;
    if (_input->Data == nullptr) {
        std::cout << "Null Input";
        exit(0);
    }
    // Release last round data
    input->release();
    preAct->release();
    output->release();

    input = new gpuMat(*_input);
    gpuMat *Wx = MatMul(weight, input);
    preAct = MatAdd(Wx, bias);
    output = activation->forwardPass(preAct);

    // Release tmp data
    Wx->release();

    return output;
}


void Layer_fc::backProp(const gpuMat* delta_out) {
    // Compute delta function
    gpuMat* dPreAct = activation->derivative(preAct);
    delta = MatEleMul(delta_out, dPreAct);

    // grad_w = delta * input^T, grad_b = delta
    gpuMat* input_t = Transpose(input);
    gpuMat* new_grad_w = MatMul(delta, input_t);
    gpuMat* new_grad_b = new gpuMat(*delta);
    if (grad_weight->Data == nullptr) {
        grad_weight = new gpuMat(weight->rows, weight->cols, weight->channels);
    }
    if (grad_bias->Data == nullptr) {
        grad_bias = new gpuMat(bias->rows, bias->cols, bias->channels);
    }

    gpuMat* acc_grad_w = MatAdd(grad_weight, new_grad_w);
    gpuMat* acc_grad_b = MatAdd(grad_bias, new_grad_b);

    cudaMemcpy(grad_weight->Data, acc_grad_w->Data, grad_weight->getSize() * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(grad_bias->Data, acc_grad_b->Data, grad_bias->getSize() * sizeof(float), cudaMemcpyDeviceToDevice);

    // Release tmp data
    dPreAct->release();
    input_t->release();
    new_grad_w->release();
    new_grad_b->release();
    acc_grad_w->release();
    acc_grad_b->release();
}

void Layer_fc::backProp(Layer_fc *next_layer) {
    // Compute delta function
    gpuMat* w_next_t = Transpose(next_layer->weight);
    gpuMat* delta_next = next_layer->delta;

    gpuMat* next_w_delta = MatMul(w_next_t, delta_next);
    gpuMat* dPreAct = activation->derivative(preAct);
    delta = MatEleMul(next_w_delta, dPreAct);

    // grad_w = delta * input^T, grad_b = delta
    if (grad_weight->Data == nullptr) {
        grad_weight = new gpuMat(weight->rows, weight->cols, weight->channels);
    }
    if (grad_bias->Data == nullptr) {
        grad_bias = new gpuMat(bias->rows, bias->cols, bias->channels);
    }
    gpuMat* input_t = Transpose(input);
    gpuMat* new_grad_w = MatMul(delta, input_t);
    gpuMat* new_grad_b = new gpuMat(*delta);
    gpuMat* acc_grad_w = MatAdd(grad_weight, new_grad_w);
    gpuMat* acc_grad_b = MatAdd(grad_bias, new_grad_b);
    cudaMemcpy(grad_weight->Data, acc_grad_w->Data, grad_weight->getSize() * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(grad_bias->Data, acc_grad_b->Data, grad_bias->getSize() * sizeof(float), cudaMemcpyDeviceToDevice);

    // Release tmp data
    input_t->release();
    new_grad_w->release();
    new_grad_b->release();
    acc_grad_w->release();
    acc_grad_b->release();
    w_next_t->release();
    delta_next->release();
    next_layer->delta->release();
    next_w_delta->release();
    dPreAct->release();
}

void Layer_fc::update(float lr) {

    // weight = MatSub(weight, MatEleMul(grad_weight, lr))
    gpuMat* new_grad_w = MatEleMul(grad_weight, lr/n_samples);
    gpuMat* a_last_grad_w = MatEleMul(last_grad_w, 0.9);
    gpuMat* b_new_grad_w = MatEleMul(new_grad_w, 0.1);
    gpuMat* grad_w = MatAdd(a_last_grad_w, b_new_grad_w);
    gpuMat* new_w = MatSub(weight, grad_w);
    cudaMemcpy(weight->Data, new_w->Data, weight->getSize() * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(last_grad_w->Data, grad_w->Data, last_grad_w->getSize() * sizeof(float), cudaMemcpyDeviceToDevice);

    //bias = MatSub(bias, MatEleMul(grad_bias, lr))
    gpuMat* new_grad_b = MatEleMul(grad_bias, lr/n_samples);
    gpuMat* a_last_grad_b = MatEleMul(last_grad_b, 0.9);
    gpuMat* b_new_grad_b = MatEleMul(new_grad_b, 0.1);
    gpuMat* grad_b = MatAdd(a_last_grad_b, b_new_grad_b);
    gpuMat* new_b = MatSub(bias, grad_b);
    cudaMemcpy(bias->Data, new_b->Data, bias->getSize() * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(last_grad_b->Data, grad_b->Data, last_grad_b->getSize() * sizeof(float), cudaMemcpyDeviceToDevice);

    // Release tmp data
    new_grad_w->release();
    a_last_grad_w->release();
    b_new_grad_w->release();
    grad_w->release();
    new_w->release();
    new_grad_b->release();
    a_last_grad_b->release();
    b_new_grad_b->release();
    grad_b->release();
    new_b->release();

    // Release iteration data
    input->release();
    preAct->release();
    output->release();
    grad_bias->release();
    grad_weight->release();
    delta->release();
    n_samples=0;

}







