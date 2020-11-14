#include "mat.h"
#include "mat_math.h"

#ifndef ACT_H
#define ACT_H

// Abstract Class
class ActivationFunction{
public:
    virtual gpuMat* forwardPass(gpuMat *preAct) = 0;
    virtual gpuMat* derivative(gpuMat *preAct) = 0;
};

class Act_Sigmoid: public ActivationFunction {
public:
    gpuMat* forwardPass(gpuMat *preAct) override;
    gpuMat* derivative(gpuMat *preAct) override;
};

class Act_ReLU: public ActivationFunction {
public:
    gpuMat* forwardPass(gpuMat *preAct) override;
    gpuMat* derivative(gpuMat *preAct) override;
};

class Act_NONE: public ActivationFunction {
public:
    gpuMat* forwardPass(gpuMat *preAct) override;
    gpuMat* derivative(gpuMat *preAct) override;
};


/*
class Act_Softmax: public ActivationFunction {
public:
    gpuMat* forwardPass(gpuMat *preAct) override;
    gpuMat* derivative(gpuMat *preAct) override;
};
*/


#endif
