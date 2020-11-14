
#ifndef MAT_H
#define MAT_H

class gpuMat;
class cpuMat;

class gpuMat {
public:
    /* Constructor and Destructor */
    gpuMat();
    gpuMat(int h, int w, int c);
    gpuMat(const gpuMat&);
    gpuMat(const cpuMat&);
    ~gpuMat();


    /* Member data */
    int cols;
    int rows;
    int channels;
    float *Data;

    /* Utils function */
    static gpuMat* ones(int rows, int cols, int channels);
    void copyTo(gpuMat &m);
    void release();
    void mallocDevice();
    void randn();
    int getSize() const;
    void print() const;
    void printVec() const;
    void print(int verbose) const;
    void printSize() const;
};

void assign_gpuMat(gpuMat* dst, gpuMat* src);


class cpuMat {
public:
    /* Constructor and Destructor */
    cpuMat();
    cpuMat(int h, int w, int c);
    cpuMat(const gpuMat&);
    cpuMat(const cpuMat&);
    ~cpuMat();

    /* Member data */
    int cols;
    int rows;
    int channels;
    float *Data;

    /* Utils function */
    static cpuMat* ones(int rows, int cols, int channels);
    void release();
    void mallocHost();
    void randn();
    int getSize() const;
    void printVec() const;
    void print();
    void printSize() const;
};

#endif // MAT_H
