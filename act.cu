#include "mat.h"
#include "mat_math.h"
#include "act.h"

////////////////
/// Sigmoid  ///
////////////////
GpuMat* Act_Sigmoid::forwardPass(GpuMat *preAct){
    return Sigmoid(preAct);
}
GpuMat* Act_Sigmoid::derivative(GpuMat *preAct){
    return dSigmoid(preAct);
}


////////////
/// ReLU ///
////////////
GpuMat* Act_ReLU::forwardPass(GpuMat *preAct){
    return ReLU(preAct);
}
GpuMat* Act_ReLU::derivative(GpuMat *preAct){
    return dReLU(preAct);
}

////////////
/// None ///
////////////
GpuMat* Act_NONE::forwardPass(GpuMat *preAct){
    return GpuMat::copy(preAct);
}
GpuMat* Act_NONE::derivative(GpuMat *preAct){
    return GpuMat::ones(preAct->rows, preAct->cols, preAct->channels);
}

