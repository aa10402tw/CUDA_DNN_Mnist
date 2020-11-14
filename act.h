#include "mat.h"
#include "mat_math.h"

#ifndef ACT_H
#define ACT_H

// Abstract Class for activation function
class ActivationFunction{
public:
    virtual GpuMat* forwardPass(GpuMat *preAct) = 0;
    virtual GpuMat* derivative(GpuMat *preAct) = 0;
};


// Sigmoid Activation function
class Act_Sigmoid: public ActivationFunction {
public:
    GpuMat* forwardPass(GpuMat *preAct) override;
    GpuMat* derivative(GpuMat *preAct) override;
};

// ReLU Activation function
class Act_ReLU: public ActivationFunction {
public:
    GpuMat* forwardPass(GpuMat *preAct) override;
    GpuMat* derivative(GpuMat *preAct) override;
};

// No Activation function (usually for last layer)
class Act_NONE: public ActivationFunction {
public:
    GpuMat* forwardPass(GpuMat *preAct) override;
    GpuMat* derivative(GpuMat *preAct) override;
};


/*
class Act_Softmax: public ActivationFunction {
public:
    GpuMat* forwardPass(GpuMat *preAct) override;
    GpuMat* derivative(GpuMat *preAct) override;
};
*/


#endif
