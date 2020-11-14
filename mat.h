#ifndef MAT_H
#define MAT_H

class GpuMat;
class CpuMat;

// Matrix class that operate data in GPU
class GpuMat {
public:
    /* Constructor and Destructor */
    GpuMat();
    GpuMat(int h, int w, int c);
    GpuMat(const GpuMat&);
    GpuMat(const CpuMat&);
    ~GpuMat();

    /* Member data */
    int cols;
    int rows;
    int channels;
    float *Data;

    /* Utils function */
    static GpuMat* zeros(int rows, int cols, int channels);
    static GpuMat* ones(int rows, int cols, int channels);
    static GpuMat* copy(const CpuMat* A);
    static GpuMat* copy(const GpuMat* A);
    void copyTo(GpuMat &m);
    void release();
    void mallocDevice();
    void randn();
    int getSize() const;
    void print() const;
    void print(int verbose) const;
    void printVec() const;
    void printSize() const;
};

void assign_gpuMat(GpuMat* dst, GpuMat* src);

// Matrix class that operate data in CPU
class CpuMat {
public:
    /* Constructor and Destructor */
    CpuMat();
    CpuMat(int h, int w, int c);
    CpuMat(const GpuMat&);
    CpuMat(const CpuMat&);
    ~CpuMat();

    /* Member data */
    int cols;
    int rows;
    int channels;
    float *Data;

    /* Utils function */
    static CpuMat* zeros(int rows, int cols, int channels);
    static CpuMat* ones(int rows, int cols, int channels);
    static CpuMat* copy(const CpuMat* A);
    static CpuMat* copy(const GpuMat* A);
    void release();
    void mallocHost();
    void randn();
    int getSize() const;
    void printVec() const;
    void print() const;
    void printSize() const;
};

#endif // MAT_H
