#include "mat.h"
#include "mat_math.h"
#include "act.h"


gpuMat* Act_Sigmoid::forwardPass(gpuMat *preAct){
    return Sigmoid(preAct);
}
gpuMat* Act_Sigmoid::derivative(gpuMat *preAct){
    return dSigmoid(preAct);
}

gpuMat* Act_ReLU::forwardPass(gpuMat *preAct){
    return ReLU(preAct);
}
gpuMat* Act_ReLU::derivative(gpuMat *preAct){
    return dReLU(preAct);
}


gpuMat* Act_NONE::forwardPass(gpuMat *preAct){
    return new gpuMat(*preAct);
}
gpuMat* Act_NONE::derivative(gpuMat *preAct){
    return gpuMat::ones(preAct->rows, preAct->cols, preAct->channels);
}

