#include "mat.h"
#include <iostream>
#include <cstdlib>
#include <iomanip>
using namespace std;


///////////////////////
/// GpuMat Function ///
///////////////////////
GpuMat::GpuMat(){
	rows = 0;
	cols = 0;
	channels = 0;
	Data = NULL;
}

GpuMat::GpuMat(int h, int w, int c) {
    cols = w;
    rows = h;
    channels = c;
    Data = NULL;
    mallocDevice();
}

GpuMat::GpuMat(const GpuMat &m) {
    cols = m.cols;
    rows = m.rows;
    channels = m.channels;
    Data = NULL;
    mallocDevice();
    cudaMemcpy(Data, m.Data, getSize() * sizeof(float), cudaMemcpyDeviceToDevice);
}


GpuMat::GpuMat(const CpuMat &m){
    cols = m.cols;
    rows = m.rows;
    channels = m.channels;
    Data = NULL;
    mallocDevice();
    cudaMemcpy(Data, m.Data, getSize() * sizeof(float), cudaMemcpyHostToDevice);
}

GpuMat* GpuMat::copy(const CpuMat* A) {
    return new GpuMat(*A);
}

GpuMat* GpuMat::copy(const GpuMat* A) {
    return new GpuMat(*A);
}

void GpuMat::copyTo(GpuMat &m) {
    cols = m.cols;
    rows = m.rows;
    channels = m.channels;
    Data = NULL;
    mallocDevice();
    cudaMemcpy(Data, m.Data, getSize() * sizeof(float), cudaMemcpyHostToDevice);
}

GpuMat::~GpuMat(){
    cudaFree(Data);
}

int GpuMat::getSize() const{
    return rows * cols * channels;
}

void GpuMat::mallocDevice() {
    if (Data==NULL) {
        cudaMalloc(&Data, sizeof(float) * getSize());
		cudaMemset(Data, 0, sizeof(float) * getSize());
    }
}

void GpuMat::release() {
    if (Data != NULL)
        cudaFree(Data);
    rows = cols = channels = 0;
    Data = nullptr;
}

void GpuMat::randn() {
    if (Data != nullptr) {
        CpuMat cm(*this);
        cm.randn();
        cudaMemcpy(Data, cm.Data, getSize() * sizeof(float), cudaMemcpyHostToDevice);
    }
}


GpuMat* GpuMat::zeros(int rows, int cols, int channels) {
    CpuMat* cm = CpuMat::zeros(rows, cols, channels);
    GpuMat* gm = GpuMat::copy(cm);
    delete cm;
    return gm;
}

GpuMat* GpuMat::ones(int rows, int cols, int channels) {
    CpuMat* cm = CpuMat::ones(rows, cols, channels);
    GpuMat* gm = GpuMat::copy(cm);
    delete cm;
    return gm;
}

void GpuMat::print() const{
    int default_verbose = 0;
    print(default_verbose);
}

void GpuMat::printVec() const {
    CpuMat cm(*this);
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

void GpuMat::print(int verbose) const {
    if(verbose > 1)
        std::cout<<"[GPU Matrix] with "<<channels<<" channels, "<<rows<<" rows, "<<cols<<" columns."<<std::endl;
    CpuMat cm(*this);
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


void GpuMat::printSize() const {
    std::cout<<"("<<channels<<","<<rows<<","<<cols<<")";
}


void assign_GpuMat(GpuMat* dst, GpuMat* src) {
    dst->release();
    src->copyTo(*dst);
}



///////////////////////
/// CpuMat Function ///
///////////////////////
CpuMat::CpuMat(){
	rows = 0;
	cols = 0;
	channels = 0;
	Data = NULL;
}

CpuMat::CpuMat(int h, int w, int c) {
    cols = w;
    rows = h;
    channels = c;
    Data = NULL;
    mallocHost();
}

CpuMat::CpuMat(const CpuMat &m) {
    cols = m.cols;
    rows = m.rows;
    channels = m.channels;
    Data = NULL;
    mallocHost();
    memcpy(Data, m.Data, getSize() * sizeof(float));
}

CpuMat::CpuMat(const GpuMat &m) {
    cols = m.cols;
    rows = m.rows;
    channels = m.channels;
    Data = NULL;
    mallocHost();
    cudaMemcpy(Data, m.Data, getSize() * sizeof(float), cudaMemcpyDeviceToHost);
}

CpuMat::~CpuMat(){
    free(Data);
}

CpuMat* CpuMat::zeros(int rows, int cols, int channels) {
    CpuMat* cm = new CpuMat(rows, cols, channels);
    int size = cm->getSize();
    for (int idx=0; idx<size; idx++) {
        cm->Data[idx] = 0.0;
    }
    return cm;
}

CpuMat* CpuMat::ones(int rows, int cols, int channels) {
    CpuMat* cm = new CpuMat(rows, cols, channels);
    int size = cm->getSize();
    for (int idx=0; idx<size; idx++) {
        cm->Data[idx] = 1.0;
    }
    return cm;
}

CpuMat* CpuMat::copy(const CpuMat* A) {
    return new CpuMat(*A);
}

CpuMat* CpuMat::copy(const GpuMat* A) {
    return new CpuMat(*A);
}

int CpuMat::getSize() const{
    return rows * cols * channels;
}

void CpuMat::mallocHost() {
    if (Data==NULL) {
        Data = (float*) malloc(sizeof(float) * getSize());
		if(NULL == Data) {
			std::cout<<"host memory allocation failed..."<<std::endl;
			exit(0);
		}
		memset(Data, 0, sizeof(float)*getSize());
    }
}

void CpuMat::release() {
    if (Data != NULL)
        free(Data);
    rows = cols = channels = 0;
    Data = nullptr;
}

void CpuMat::randn() {
    if (Data != nullptr) {
        int i = 0;
        while (i < getSize()) {
            Data[i] = (0.5f - float(rand()) / float(RAND_MAX))/10.0;
            i += 1;
        }
    }
}

void CpuMat::print() const{
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

void CpuMat::printVec() const {
    CpuMat cm(*this);
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

void CpuMat::printSize() const {
    std::cout<<"("<<channels<<","<<rows<<","<<cols<<")";
}

