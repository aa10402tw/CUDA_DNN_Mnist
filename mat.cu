#include "mat.h"
#include <iostream>
#include <cstdlib>
#include <iomanip>
using namespace std;


// GPU Mat
gpuMat::gpuMat(){
	rows = 0;
	cols = 0;
	channels = 0;
	Data = NULL;
}

gpuMat::gpuMat(int h, int w, int c) {
    cols = w;
    rows = h;
    channels = c;
    Data = NULL;
    mallocDevice();
}

gpuMat::gpuMat(const gpuMat &m) {
    cols = m.cols;
    rows = m.rows;
    channels = m.channels;
    Data = NULL;
    mallocDevice();
    cudaMemcpy(Data, m.Data, getSize() * sizeof(float), cudaMemcpyDeviceToDevice);
}


gpuMat::gpuMat(const cpuMat &m){
    cols = m.cols;
    rows = m.rows;
    channels = m.channels;
    Data = NULL;
    mallocDevice();
    cudaMemcpy(Data, m.Data, getSize() * sizeof(float), cudaMemcpyHostToDevice);
}

void gpuMat::copyTo(gpuMat &m) {
    cols = m.cols;
    rows = m.rows;
    channels = m.channels;
    Data = NULL;
    mallocDevice();
    cudaMemcpy(Data, m.Data, getSize() * sizeof(float), cudaMemcpyHostToDevice);
}

gpuMat::~gpuMat(){
    cudaFree(Data);
}

int gpuMat::getSize() const{
    return rows * cols * channels;
}

void gpuMat::mallocDevice() {
    if (Data==NULL) {
        cudaMalloc(&Data, sizeof(float) * getSize());
		cudaMemset(Data, 0, sizeof(float) * getSize());
    }
}

void gpuMat::release() {
    if (Data != NULL)
        cudaFree(Data);
    rows = cols = channels = 0;
    Data = nullptr;
}

void gpuMat::randn() {
    if (Data != nullptr) {
        cpuMat cm(*this);
        cm.randn();
        cudaMemcpy(Data, cm.Data, getSize() * sizeof(float), cudaMemcpyHostToDevice);
    }
}

gpuMat* gpuMat::ones(int rows, int cols, int channels) {
    cpuMat* cm = cpuMat::ones(rows, cols, channels);
    gpuMat* gm = new gpuMat(*cm);
    return gm;
}

void gpuMat::print() const{
    int default_verbose = 0;
    print(default_verbose);
}

void gpuMat::printVec() const {
    cpuMat cm(*this);
    cout<< setprecision(3) << setiosflags(ios::fixed);
    cout << "[";
    for (int c=0; c < getSize(); c++) {
        if (cm.Data[c] < 0) cout<< setprecision(3) << setiosflags(ios::fixed) << cm.Data[c];
        else cout<< setprecision(4) << setiosflags(ios::fixed) << cm.Data[c];
        if (c < getSize() - 1)
            cout << " ";
    }
    cout << "] ";
    printSize();
    cout << endl;
}

void gpuMat::print(int verbose) const {
    if(verbose > 1)
        std::cout<<"[GPU Matrix] with "<<channels<<" channels, "<<rows<<" rows, "<<cols<<" columns."<<std::endl;
    cpuMat cm(*this);
    int cnt = 0;
    for (int c=0; c<channels; c++) {
        if(verbose > 0)
            std::cout << "Channel: " << c  << std::endl;
        for (int r=0; r<rows; r++) {
            for (int c=0; c<cols; c++) {
                std::cout << cm.Data[cnt] << " ";
                cnt += 1;
            }
            std::cout << std::endl;
        }
    }
}


void gpuMat::printSize() const {
    std::cout<<"("<<channels<<","<<rows<<","<<cols<<")";
}


void assign_gpuMat(gpuMat* dst, gpuMat* src) {
    dst->release();
    src->copyTo(*dst);
}


// CPU Mat
cpuMat::cpuMat(){
	rows = 0;
	cols = 0;
	channels = 0;
	Data = NULL;
}

cpuMat::cpuMat(int h, int w, int c) {
    cols = w;
    rows = h;
    channels = c;
    Data = NULL;
    mallocHost();
}

cpuMat::cpuMat(const cpuMat &m) {
    cols = m.cols;
    rows = m.rows;
    channels = m.channels;
    Data = NULL;
    mallocHost();
    memcpy(Data, m.Data, getSize() * sizeof(float));
}

cpuMat::cpuMat(const gpuMat &m) {
    cols = m.cols;
    rows = m.rows;
    channels = m.channels;
    Data = NULL;
    mallocHost();
    cudaMemcpy(Data, m.Data, getSize() * sizeof(float), cudaMemcpyDeviceToHost);
}

cpuMat::~cpuMat(){
    free(Data);
}

cpuMat* cpuMat::ones(int rows, int cols, int channels) {
    cpuMat* cm = new cpuMat(rows, cols, channels);
    int size = cm->getSize();
    for (int idx=0; idx<size; idx++) {
        cm->Data[idx] = 1.0;
    }
    return cm;
}

int cpuMat::getSize() const{
    return rows * cols * channels;
}

void cpuMat::mallocHost() {
    if (Data==NULL) {
        Data = (float*) malloc(sizeof(float) * getSize());
		if(NULL == Data) {
			std::cout<<"host memory allocation failed..."<<std::endl;
			exit(0);
		}
		memset(Data, 0, sizeof(float)*getSize());
    }
}

void cpuMat::release() {
    if (Data != NULL)
        free(Data);
    rows = cols = channels = 0;
    Data = nullptr;
}

void cpuMat::randn() {
    if (Data != nullptr) {
        int i = 0;
        while (i < getSize()) {
            Data[i] = 0.5f - float(rand()) / float(RAND_MAX);
            i += 1;
        }
    }
}

void cpuMat::print() {
    std::cout<<"[CPU Matrix] with "<<channels<<" channels, "<<rows<<" rows, "<<cols<<" columns."<<std::endl;
    int cnt = 0;
    for (int c=0; c<channels; c++) {
        std::cout << "Channel: " << c  << std::endl;
        for (int r=0; r<rows; r++) {
            for (int c=0; c<cols; c++) {
                std::cout << Data[cnt] << " ";
                cnt += 1;
            }
            std::cout << std::endl;
        }
    }
}

void cpuMat::printVec() const {
    cpuMat cm(*this);
    cout<< setprecision(3) << setiosflags(ios::fixed);
    cout << "[";
    for (int c=0; c < getSize(); c++) {
        if (cm.Data[c] < 0) cout<< setprecision(3) << setiosflags(ios::fixed) << cm.Data[c];
        else cout<< setprecision(4) << setiosflags(ios::fixed) << cm.Data[c];
        if (c < getSize() - 1)
            cout << " ";
    }
    cout << "] ";
    printSize();
    cout << endl;
}

void cpuMat::printSize() const {
    std::cout<<"("<<channels<<","<<rows<<","<<cols<<")";
}

