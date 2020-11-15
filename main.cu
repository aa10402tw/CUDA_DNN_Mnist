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
GpuMat* dCrossEntropyLoss(const GpuMat *output, const GpuMat* target);
GpuMat* imgToInput(MNIST_DATA_TYPE data[28][28]);
GpuMat* labelToTarget(unsigned int label);
int argmax(CpuMat* onehot);
bool isCorrect(const GpuMat* target, const GpuMat* output);
GpuMat* fowardPass(std::vector<Layer_fc*> &net, const GpuMat* input);
void backProp(std::vector<Layer_fc*> &net, GpuMat* dLoss);
void update(std::vector<Layer_fc*> &net, float lr);

void train(std::vector<Layer_fc*> &net, int batch_size, float lr, int log_intervel);
void test(std::vector<Layer_fc*> &net);

// Load Mnist data
static inline void loaddata() {
	mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);
}

// Print Network layers
void PrintNet(std::vector<Layer_fc*> &net) {
    for(int i=0; i<net.size(); i++) {
        std::cout << "[Layer: " << i << "]";
        std::cout << " W: ";
        net[i]->weight->printSize();
        std::cout << ", b: ";
        net[i]->bias->printSize();
        std::cout << std::endl;
    }
}

// Compute the deriviate of CrossEntropyLoss [dCE(y_pred, y_true) = y_pred - y_true]
GpuMat* dCrossEntropyLoss(const GpuMat *output, const GpuMat* target) {
    return MatSub(softmax(output), target);
}

// CrossEntropyLoss [dCE(y_pred, y_true) = -log(y_pred[idx]) for correct idx]
float crossEntropyLoss(GpuMat* output, GpuMat* target) {
    CpuMat* sm = new CpuMat(*softmax(output));
    int idx = argmax(new CpuMat(*target));
    return -log (sm->Data[idx]);
}

// Print the image label and data 
void printImage(mnist_data d, int scale_down) {
    std::cout << "Label:" << d.label << std::endl;
    for (int i=0; i<28; i+=scale_down) {
        for (int j=0; j<28; j+=scale_down) {
            if (d.data[i][j] > 0.5) std::cout << "*";
            else std::cout << " ";
        }
        std::cout << std::endl;
    }
}

void printImage(mnist_data d) {
    printImage(d, 2);
}

// Convert image data to flattern GpuMat
GpuMat* imgToInput(MNIST_DATA_TYPE data[28][28]) {
    CpuMat* input_cpu = new CpuMat(28*28, 1, 1);
    for (int i=0; i<28; i++)
        for (int j=0; j<28; j++)
            input_cpu->Data[i*28+j] = data[i][j];
    return new GpuMat(*input_cpu);
}

// Convert the unsigned int label to onehot vector
GpuMat* labelToTarget(unsigned int label) {
    CpuMat* target_cpu = new CpuMat(10, 1, 1);
    target_cpu->Data[label] = 1.0;
    return new GpuMat(*target_cpu);
}

// Get the argmax (index of max value) of onehot vector
int argmax(CpuMat* onehot) {
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

// Return true if y_pred = y_true 
bool isCorrect(const GpuMat* output, const GpuMat* target) {
    int y_pred = argmax(new CpuMat(*output));
    int y_true = argmax(new CpuMat(*target));
    return (y_pred == y_true);
}

// Forward-pass the net
GpuMat* fowardPass(std::vector<Layer_fc*> &net, const GpuMat* input) {
    GpuMat* output = nullptr;
    for(int i=0; i<net.size(); i++) {
        output = net[i]->forwardPass(input);
        input = output;
    }
    return new GpuMat(*output);
}

// Back-propgation for the last layer
void backProp(std::vector<Layer_fc*> &net, GpuMat* dLoss) {
    for(int i=net.size()-1; i>=0; i--) {
        if (i == net.size()-1) {
            net[i]->backProp(dLoss);
        }
        else {
            net[i]->backProp(net[i+1]);
        }
    }
}

// Update the weight of net
void update(std::vector<Layer_fc*> &net, float lr) {
    for(int i=0; i<net.size(); i++) {
        net[i]->update(lr);
    }
}

// Reset the gradient of net
void reset_grad(std::vector<Layer_fc*> &net) {
    for(int i=0; i<net.size(); i++) {
        net[i]->reset_grad();
    }
}


// Training for 1 epoch
void train(std::vector<Layer_fc*> &net, int batch_size, float lr, int log_intervel) {
    srand( (int)time(NULL)*1000 );
    std::vector<int> idxes(train_cnt);
    for(int i=0; i<train_cnt; i++)
        idxes[i] = i;
    std::random_shuffle ( idxes.begin(), idxes.end() );

    std::cout << "\n[Training Start]" << std::endl;
    int n_correct = 0;
    double loss_total = 0.0;
    for (int it=0; it<train_cnt/batch_size; it++) {
        // Train a batch
        for(int i=0; i<batch_size; i++) {
            int idx = idxes[it*batch_size+i];
            GpuMat* input = imgToInput(train_set[idx].data);
            GpuMat* target = labelToTarget(train_set[idx].label);
            GpuMat* output = fowardPass(net, input);
            GpuMat* dLoss = dCrossEntropyLoss(output, target);
            backProp(net, dLoss);

            if (isCorrect(output, target)) n_correct += 1;
            loss_total += crossEntropyLoss(output, target);
        }
        // Update
        update(net, lr);

        // Log
        if (log_intervel > 0 && it % log_intervel == 0) {
            int cnt = (it+1)*batch_size;
            std::cout << "iter:" << it << ", loss:" << loss_total / (float)(cnt) << ", acc=" <<  (float)n_correct / (float)(cnt) << std::endl;
        }
    }

    std::cout << "Train Accuracy: " << (float)n_correct/(float)train_cnt << std::endl; 
    std::cout << "Train Loss: " << loss_total/(double)train_cnt << std::endl; 
    reset_grad(net);
}

// Test network
void test(std::vector<Layer_fc*> &net) {
    std::cout << "[Test Start]" << std::endl;
    int n_correct = 0;
    double loss_total = 0.0;
    for (int i=0; i<test_cnt; i++) {
        GpuMat* input = imgToInput(test_set[i].data);
        GpuMat* target = labelToTarget(test_set[i].label);
        GpuMat* output = fowardPass(net, input);

        if (isCorrect(output, target)) n_correct += 1;
        loss_total += crossEntropyLoss(output, target);

        if ( (i+1) % (test_cnt/10) == 0 ) {
            int progress = (float)(i+1) / (float)(test_cnt/10) * 10.0;
            std::cout << progress << "% ";
        } 
    }
    std::cout << "Test Accuracy: " << (float)n_correct/(float)test_cnt << std::endl; 
    std::cout << "Test Loss: " << loss_total/(double)test_cnt << std::endl; 
    reset_grad(net);
}


int main() {
    // Load Data for minist
    loaddata();

    // Peek the dataset
    for (int i=0; i<5; i++)
        printImage(train_set[i]);

    // Define Layer architceture for fully-connected layer with activation function
    std::vector<Layer_fc*> net;
    net.push_back(new Layer_fc(28*28,  1024, (ActivationFunction*) new Act_ReLU()));
    net.push_back(new Layer_fc(1024,   512, (ActivationFunction*) new Act_ReLU()));
    net.push_back(new Layer_fc(512,    256, (ActivationFunction*) new Act_ReLU()));
    net.push_back(new Layer_fc(256,     10, (ActivationFunction*) new Act_NONE()));

    // Define the hyper parameters and log interval (-1 for no log)
    int batch_size = 1;
    float lr = 0.01;
    int log_intervel = 10000 / batch_size;
    int epochs = 3;

    // Training the network
    for (int i=0; i<epochs; i++)
        train(net, batch_size, lr, log_intervel);

    // Test the network
    test(net);
}
