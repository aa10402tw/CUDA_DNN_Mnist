#include <time.h>
#include <vector>
#include <iostream>
#include <algorithm>    // std::random_shuffle
#include <math.h>

#include "layer_fc.h"
#include "mat.h"
#include "mat_math.h"
#include "act.h"

#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

static inline void loaddata();
gpuMat* dCrossEntropyLoss(const gpuMat *output, const gpuMat* target);
void train(std::vector<Layer_fc*> &net);
gpuMat* imgToInput(MNIST_DATA_TYPE data[28][28]);
gpuMat* labelToTarget(unsigned int label);
int argmax(cpuMat* onehot);
bool isCorrect(const gpuMat* target, const gpuMat* output);
gpuMat* fowardPass(std::vector<Layer_fc*> &net, const gpuMat* input);
void backProp(std::vector<Layer_fc*> &net, gpuMat* dLoss);
void update(std::vector<Layer_fc*> &net, float lr);

static inline void loaddata()
{
	mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);
}


void PrintNet(std::vector<Layer_fc*> &net) {
    /// Print Network layers
    for(int i=0; i<net.size(); i++) {
        std::cout << "[Layer: " << i << "]";
        std::cout << " W: ";
        net[i]->weight->printSize();
        std::cout << ", b: ";
        net[i]->bias->printSize();
        std::cout << std::endl;
    }
}


gpuMat* dCrossEntropyLoss(const gpuMat *output, const gpuMat* target) {
    return MatSub(softmax(output), target);
}

float crossEntropyLoss(gpuMat* output, gpuMat* target) {
    cpuMat* sm = new cpuMat(*softmax(output));
    int idx = argmax(new cpuMat(*target));
    return -log (sm->Data[idx]);
}


void printImage(mnist_data d) {
    std::cout << "Label:" << d.label << std::endl;
    for (int i=0; i<28; i++) {
        for (int j=0; j<28; j++) {
            if (d.data[i][j] > 0.5) std::cout << "*";
            else std::cout << " ";
        }
        std::cout << std::endl;
    }

}

gpuMat* imgToInput(MNIST_DATA_TYPE data[28][28]) {
    cpuMat* input_cpu = new cpuMat(28*28, 1, 1);
    for (int i=0; i<28; i++)
        for (int j=0; j<28; j++)
            input_cpu->Data[i*28+j] = data[i][j];
    return new gpuMat(*input_cpu);
}

gpuMat* labelToTarget(unsigned int label) {
    cpuMat* target_cpu = new cpuMat(10, 1, 1);
    target_cpu->Data[label] = 1.0;
    return new gpuMat(*target_cpu);
}

int argmax(cpuMat* onehot) {
    int idx = 0;
    float maxVal = onehot->Data[0];
    for (int i=1; i<onehot->getSize(); i++) {
        if ( onehot->Data[i] > maxVal) {
            idx = i;
            maxVal = onehot->Data[i];
        }
    }
    return idx;
}

bool isCorrect(const gpuMat* target, const gpuMat* output) {
    int y_pred = argmax(new cpuMat(*output));
    int y_true = argmax(new cpuMat(*target));
    return (y_pred == y_true);
}

gpuMat* fowardPass(std::vector<Layer_fc*> &net, const gpuMat* input) {
    gpuMat* output = nullptr;
    for(int i=0; i<net.size(); i++) {
        output = net[i]->forwardPass(input);
        input = output;
    }
    return new gpuMat(*output);
}

void backProp(std::vector<Layer_fc*> &net, gpuMat* dLoss) {
    for(int i=net.size()-1; i>=0; i--) {
        if (i == net.size()-1) {
            net[i]->backProp(dLoss);
        }
        else {
            net[i]->backProp(net[i+1]);
        }
    }
}

void update(std::vector<Layer_fc*> &net, float lr) {
    for(int i=0; i<net.size(); i++) {
        net[i]->update(lr);
    }
}


void train(std::vector<Layer_fc*> &net, int batch_size, float lr) {
    std::cout << "Training Start:\n";

    int n_correct = 0;
    double loss_total = 0.0;
    std::vector<int> idxes(train_cnt);
    for(int i=0; i<train_cnt; i++)
        idxes[i] = i;

    for (int i=0; i<5; i++)
        printImage(train_set[idxes[i]]);

    std::random_shuffle ( idxes.begin(), idxes.end() );
    for (int it=0; it<train_cnt/batch_size; it++) {
        // Train a batch
        double loss_batch = 0.0;
        for(int i=0; i<batch_size; i++) {
            int idx = idxes[it*batch_size+i];
            gpuMat* input = imgToInput(train_set[idx].data);
            gpuMat* target = labelToTarget(train_set[idx].label);
            gpuMat* output = fowardPass(net, input);
            gpuMat* dLoss = dCrossEntropyLoss(output, target);
            backProp(net, dLoss);

            if (isCorrect(target, output))
                n_correct += 1;
            loss_batch += crossEntropyLoss(output, target);
        }
        // Update
        update(net, lr);

        // Log
        if (it % 100 == 0) {
            int cnt = (it+1)*batch_size;
            std::cout << "iter:" << it << ", loss:" << loss_batch / (float)(batch_size) << ", acc=" <<  (float)n_correct / (float)(cnt) << std::endl;
        }

    }
}

int main() {
    srand( (int)time(NULL)*1000 );
    loaddata();

    std::vector<Layer_fc*> net;
    net.push_back(new Layer_fc(28*28, 200, (ActivationFunction*) new Act_Sigmoid()));
    net.push_back(new Layer_fc(200, 80,    (ActivationFunction*) new Act_Sigmoid()));
    net.push_back(new Layer_fc(80,  10,   (ActivationFunction*) new Act_NONE()));

    //train(net, 0.1);
    int batch_size = 1;
    float lr = 0.01;
    train(net, batch_size, lr);
    train(net, batch_size, lr);
    train(net, batch_size, lr);
}
