#include <mat.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <matrix.h>
#include <iostream>
#include <algorithm>
#include "error_util.h"
#include <cuda_runtime.h>
using namespace std;

const int FILTERS_NUM = 20;
const int COV_LEN = 19;
const int STRIDE = 2;
const int POOLING_LEN = 2;
const int NEU_NUM1 = 100;
const int NEU_NUM2 = 13; 
const int NEIGHBOR = 8;
double learning_rate = 0.008;
const double MIN_ERR = 0.0003;
const int VALID_BATCH = 5;

//Initialize CUDA
bool InitCUDA(){
    int count;
    cudaGetDeviceCount(&count);
    if(count==0){
        fprintf(stderr,"There is no device.\n");
        return false;
    }
    int i;
    for (i =0; i<count;i++){
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop,i)==cudaSuccess){
            if(prop.major>=1){     
                break;
            }
        }
    }
    if(i==count){
        fprintf(stderr,"There is no device supporting CUDA 1.x.\n");
        return false;
    }
    cudaSetDevice(0);
    return true;
}

template<typename T>
struct Tensor {
    int length;
    T *data_h;
    T *data_d;

    // constructor
    Tensor();
};

template<typename T>
Tensor<T>::Tensor()
{
    length = 0;
    data_h = NULL;
    data_d = NULL;
}

struct DataLayer {
    Tensor<double> input;
    Tensor<int> labels;

    // constructor
    DataLayer(int input_size, int labels_size);
    // destructor
    ~DataLayer();
};

DataLayer::DataLayer(int input_size, int labels_size)
{
    input.length = input_size;
    labels.length = labels_size;
}

DataLayer::~DataLayer()
{
    if ( input.data_h != NULL )
        delete [] input.data_h;
    if ( labels.data_h != NULL )
        delete [] labels.data_h;
    if ( input.data_d != NULL )
        cudaFree(input.data_d);
    if ( labels.data_d != NULL )
        cudaFree(labels.data_d);
}

struct Layer {
    Tensor<double> input;
    Tensor<double> output;
    Tensor<double> weights;
    Tensor<double> bias;
    Tensor<double> deltaW;
    Tensor<double> deltaB;

    // constructor
    Layer(int input_size, int weights_size, int bias_size, int output_size, bool copyback);
    // destructor
    ~Layer();

private:
    void allocMemcpyCuda(int size, double **data_h, double **data_d, bool isMallochost, bool isInitalize);
};

void Layer::allocMemcpyCuda(int size, double **data_h, double **data_d, bool isMallochost, bool isInitalize)
{
    size_t sizeBytes = size * sizeof(double);
    checkCudaErrors(cudaMalloc(data_d, sizeBytes));

    if ( isMallochost )
    {
        *data_h = new double [size];

        if ( isInitalize )
        {
            for ( int i = 0; i < size; i ++ )
                data_h[0][i] = (2 * (rand() / double(RAND_MAX)) - 1 ) / 50;
            checkCudaErrors(cudaMemcpy(*data_d, *data_h, sizeBytes, cudaMemcpyHostToDevice));
        }
    }
}

Layer::Layer(int input_size, int weights_size, int bias_size, int output_size, bool copyback)
{
    input.length = input_size;
    weights.length = weights_size;
    deltaW.length = weights_size;
    output.length = output_size;
    bias.length = bias_size;
    deltaB.length = bias_size;

    //allocMemcpyCuda(input.length, &input.data_h, &input.data_d, false, false);
    allocMemcpyCuda(weights.length, &weights.data_h, &weights.data_d, true, true);
    allocMemcpyCuda(bias.length, &bias.data_h, &bias.data_d, true, true);
    allocMemcpyCuda(output.length, &output.data_h, &output.data_d, copyback, false);
    allocMemcpyCuda(deltaB.length, &deltaB.data_h, &deltaB.data_d, false, false);
    allocMemcpyCuda(deltaW.length, &deltaW.data_h, &deltaW.data_d, false, false);
}

Layer::~Layer()
{
    if ( input.data_h != NULL )
        delete[] input.data_h;
    if ( weights.data_h != NULL )
        delete [] weights.data_h;
    if ( output.data_h != NULL )
        delete [] output.data_h;
    if ( bias.data_h != NULL )
        delete [] bias.data_h;
    if ( deltaW.data_h != NULL )
        delete [] deltaW.data_h;
    if ( deltaB.data_h != NULL )
        delete [] deltaB.data_h;
    if ( input.data_d != NULL )
        cudaFree(input.data_d);
    if ( weights.data_d != NULL )
        cudaFree(weights.data_d);
    if ( output.data_d != NULL )
        cudaFree(output.data_d);
    if ( bias.data_d != NULL )
        cudaFree(bias.data_d);
    if ( deltaW.data_d != NULL )
        cudaFree(deltaW.data_d);
    if ( deltaB.data_d != NULL )
        cudaFree(deltaB.data_d);
}

__device__ void copy_data_to_shared(double * data, double * data_tmp, int tid, int offset, int head, int length)
{
    for(size_t i = tid * offset; i < (tid + 1) * offset && (i < length); i++){
        data_tmp[i] = data[i + head];
    }

    __syncthreads();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//forward propagation
__global__ static void convolution( int epoch_id,
                                    int perLayerSize,
                                    int cube_len,
                                    int conv_len,
                                    int filter_num,
                                    int stride,
                                    double * input,
                                    double * filters, 
                                    double * bias,
                                    double * output )
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int output_size = (cube_len - conv_len - 1) / stride + 1;

    if ( tid < output_size && bid < filter_num )
    {
        int cube_size = cube_len * perLayerSize;
        extern __shared__ double tmp[];
        int head = epoch_id * cube_size;
        int length = cube_size;
        int offset = (length - 1) / output_size + 1;
        copy_data_to_shared(input, tmp, tid, offset, head, length);
        
        int filterSize = perLayerSize * conv_len;
        head = bid * filterSize;
        length = filterSize;
        offset = (length - 1) / output_size + 1;
        copy_data_to_shared(filters, tmp + cube_size, tid, offset, head, length);
        __syncthreads();

        double mid = 0;
        for ( int i = 0; i< filterSize; i ++ ) {
            mid = mid + tmp[i + cube_size] * tmp[tid * perLayerSize * stride + i];
        }

        mid = mid + bias[bid];
        output[tid + bid * output_size] = 2/(1 + (1 / exp(2 * mid))) - 1;
    }
}

__global__ static void maxpooling( int input_size,
                                   int pooling_size,
                                   int group_num,
                                   double * input,
                                   double * output,
                                   double * output_index )
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int output_size = (input_size -1) / pooling_size + 1;

    if ( tid < output_size && bid < group_num ) 
    {
        double mid;
        int mid_index, head, tail;
        head = tid * pooling_size + bid * input_size;
        tail = (tid + 1) * pooling_size + bid * input_size ;
        mid = input[head];
        mid_index = head;
        for ( int i = head; i < tail && (i < (bid + 1) * input_size); i ++ )
        {
            if ( mid < input[i] ) {
                mid = input[i];
                mid_index=i;
            }
        }
        output[tid + bid * output_size] = mid;
        output_index[tid + bid * output_size] = mid_index;
    }
}

__global__ static void fully_connect( int input_size,
                                      int output_size,
                                      double * input,
                                      double * weights,
                                      double * bias,
                                      double * output )
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if ( tid < input_size && bid < output_size ) 
    {
        extern __shared__ double ner[];
        ner[tid] = weights[bid + tid * output_size] * input[tid];
        __syncthreads();

        int length = input_size;
        int offset = (length - 1) / 2 + 1;

        while ( length >= 2 )
        {
            if(tid + offset < length)
            {
                ner[tid] = ner[tid] + ner[tid + offset];
            }
            offset = (offset - 1)/2 + 1;
            length = (length - 1)/2 + 1;
            __syncthreads();
        }

        if ( tid < 1 )
            output[bid] = 2 / (1 + 1 / exp((ner[0] + bias[bid]) * 2)) - 1;
    }
}

__global__ static void output_and_loss( bool tag, 
                                        int train_idx,
                                        int input_size,
                                        int output_size,
                                        double * input, 
                                        double * weights, 
                                        double * bias, 
                                        double * output,
                                        double * labels,
                                        double * dValue,
                                        double * loss ) 
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if ( id < output_size )
    {
        extern __shared__ double tmp[];
        size_t offset = (input_size - 1) / output_size + 1;
        copy_data_to_shared(input, tmp, id, offset, 0, input_size);
        __syncthreads();
        
        double mid = 0;
        for ( size_t i = 0; i < input_size; i ++ ) {
            mid = mid + weights[id + i * output_size] * tmp[i];
        }
        output[id] = exp(mid+ bias[id]);
        tmp[id + input_size] = output[id];
        __syncthreads();
        
        size_t length = output_size;
        int output_offset = input_size + id;
        offset = (length - 1) / 2 +1;
        while(length >= 2)
        {
            if ( id + offset < length ) {
                tmp[output_offset] = tmp[output_offset] + tmp[output_offset + offset];
            }
            
            offset = (offset - 1)/2 + 1;
            length = (length - 1)/2 + 1;
             __syncthreads();
        }

        output[id] = output[id] / tmp[input_size];
        
        // loss function        
        if ( tag )
        {
            dValue[id] = (output[id] - labels[id + train_idx * output_size]) / output_size;

            if ( train_idx == 0 )
                loss[0] = 0;

            int loss_offset = input_size + output_size + id;
            tmp[loss_offset] = labels[id + train_idx * output_size] * log(output[id]) + 
                               (1 - labels[id + train_idx * output_size]) * log(1 - output[id]);
            __syncthreads();
        
            length = output_size;
            offset = (length - 1) / 2 + 1;

            while ( length >= 2 ) {
                if ( id + offset < length ) {
                   tmp[loss_offset] = tmp[loss_offset] + tmp[loss_offset + offset];
                }

                offset = (offset - 1)/2 + 1;
                length = (length - 1)/2 + 1;
                __syncthreads();
             }
 
            if ( id < 1 )
                loss[0] = loss[0] - tmp[input_size + output_size] / output_size;
         }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// backward propagation
__global__ static void bp_fully_connect( int input_size,
                                         int output_size,
                                         double learning_rate, 
                                         double * weights,
                                         double * bias, 
                                         double * data, 
                                         double * deltaB, 
                                         double * fol_deltaZ )
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if ( tid < output_size && bid < input_size ) {
        extern __shared__ double tmp[];
        tmp[tid] = weights[tid + bid * output_size] * deltaB[tid];
        __syncthreads();

        weights[tid + bid * output_size] = weights[tid + bid * output_size] - learning_rate * data[bid] * deltaB[tid];

        int length = output_size;
        int offset = (length - 1) / 2 + 1;

        while ( length >= 2 ) {
            if ( tid + offset < length ) {
                tmp[tid] = tmp[tid] + tmp[tid + offset];
            }
            length = (length - 1)/2 + 1;
            offset = (offset - 1)/2 + 1;
            __syncthreads();
        }

        if ( bid < 1 )
            bias[tid] = bias[tid] - learning_rate * deltaB[tid];

        if ( tid < 1 )        
            fol_deltaZ[bid] = tmp[0] * (1 + data[bid]) * (1 - data[bid]);
    }
}

__global__ static void bp_maxpooling( int output_size,
                                      double * mre_index,
                                      double * deltaB,
                                      double * fol_deltaZ )
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if ( tid < output_size )
    {
        int idx = int(mre_index[tid]);

        fol_deltaZ[idx] = deltaB[tid];
    }
}

__global__ static void bp_convolution( int epoch_id,
                                       int stride,
                                       int perLayerSize,
                                       int cube_len,
                                       int filter_size,
                                       int filter_num,
                                       int output_size, 
                                       double learning_rate,
                                       double * pre_deltaB, 
                                       double * filters, 
                                       double * bias, 
                                       double * data, 
                                       double * output )
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if ( tid < filter_size && bid < filter_num )
    {
        int re_size = output_size / filter_num;
        int cube_size = cube_len * perLayerSize;
        int head = epoch_id * cube_size;
        int length = cube_size;
        int offset = (length - 1) / filter_size + 1;
        
        extern __shared__ double tmp[];
        copy_data_to_shared(data, tmp, tid, offset, head, length);
        __syncthreads();

        double mid0 = 0, mid1 = 0;
        for ( size_t i = 0; i < re_size; i ++ ) {
            mid0 = mid0 + pre_deltaB[i + bid * re_size] * tmp[tid + i * perLayerSize * stride];
            mid1 = mid1 + pre_deltaB[i + bid * re_size] * (1 + output[i + bid * re_size]) * (1 - output[i + bid * re_size]);
        }

        //mid0 = mid0 / re_size;
        filters[tid + bid * filter_size] = filters[tid + bid * filter_size] - learning_rate * mid0 / re_size;
                
        if(tid < 1)
            bias[bid] = bias[bid] - learning_rate * (mid1 / re_size);
        
    }
}

__global__ static void preprocessing(int iter, double * data, int * train_index, double * processed_data, int x, int y, int z, int train_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int threadNum = blockDim.x * gridDim.x;
    int id = tid + iter * threadNum;

    if (id < train_size){
        int idx = id * (NEIGHBOR+1) * z;
        int i, j;
        for (i=0; i<z; i++){
            for (j=0; j<(NEIGHBOR+1); j++){
                processed_data[idx] = data[train_index[j + id*(NEIGHBOR+1)] + i * x*y];
                idx = idx + 1;    
            }
        }
    }
}

double lossfunction(double * output, double * labels, int idx){
    double l = 0;
    int i;
    for(i=0; i<NEU_NUM2; i++){
        l = l + labels[i + idx*NEU_NUM2] * log(output[i]) + (1 - labels[i + idx*NEU_NUM2])*log(1 - output[i]); 
    }
    l = -l/NEU_NUM2;
    return l;
}

// calculate accuracy rate
double count_err(double * test_labels, double * output, int test_idx)
{
    double right=0;
    double max =0;
    int idx = 0;
    int i;
    for(i=0; i<NEU_NUM2; i++){
        if(output[i]>max){
            max = output[i];
            idx = i;
        }
    }
    if((idx+1) == int(test_labels[test_idx]))
        right = 1;
    
    return right;
}


void insert_line(double * a, double b){
    for(int i=1; i<VALID_BATCH; i++){
        a[i-1] = a[i];
    }
    a[VALID_BATCH-1] = b;
}

// shuffle
void shuffle(int * data, double * labels, int dim_row, int width){
    int index,  i;
    int temp;
    double tmp;
    srand(time(NULL));
    for(i=0; i<width; i++){
        index=rand()%(width-i) + i;
        if(index != i){
            for(int j=0; j<dim_row; j++){
                temp = data[j + i*dim_row];
                data[j + i*dim_row] = data[j +index*dim_row];
                data[j + index*dim_row] = temp;
            }

            for(int j=0; j<NEU_NUM2; j++){
                tmp = labels[j + i*NEU_NUM2];
                labels[j + i*NEU_NUM2] = labels[j + index*NEU_NUM2];
                labels[j + index*NEU_NUM2] = tmp;
            }
        }
    }
}


double training(double * data, double * labels, int x, int y, int z){
    clock_t start, end;
    start = clock();
    double * gpu_data;
    double * gpu_processed_train;
    double * gpu_processed_test;
    int * gpu_train_index;
    int * gpu_test_index;
    double * gpu_processed_labels;

    int data_size = 0;
    int * data_index = new int [x*y];
    for(int i=0; i<x*y; i++){
        if(labels[i] != 0){
            data_index[data_size]=i;
            data_size ++;
        }
    }
    int test_size = (data_size-1)/5 + 1;
    int train_size = data_size - test_size;
    int * train_index = new int [train_size * (NEIGHBOR + 1)];
    int * test_index = new int [test_size * (NEIGHBOR + 1)];

    double * processed_labels = new double [train_size * NEU_NUM2]();
    double * test_labels = new double [test_size]();

    int tr=0, te=0;
    for (int i=0; i<data_size; i++){
        if (i%5 != 0){
            train_index[(NEIGHBOR/2) + tr * (NEIGHBOR+1)] = data_index[i];
            if(NEIGHBOR == 4)
            {
                train_index[(NEIGHBOR/2) + tr * (NEIGHBOR+1) - 1] = data_index[i] - 1;
                train_index[(NEIGHBOR/2) + tr * (NEIGHBOR+1) + 1] = data_index[i] + 1;
                train_index[0 + tr * (NEIGHBOR+1)] = data_index[i] - x;
                train_index[NEIGHBOR + tr * (NEIGHBOR+1)] = data_index[i] + x;
                

                if((data_index[i] % x) == 0){//first row
                    train_index[(NEIGHBOR/2) + tr * (NEIGHBOR+1) - 1] = train_index[(NEIGHBOR/2) + tr * (NEIGHBOR+1) + 1];
                }
                if((data_index[i] % x) == (x-1)){//last row
                    train_index[(NEIGHBOR/2) + tr * (NEIGHBOR+1) + 1] = train_index[(NEIGHBOR/2) + tr * (NEIGHBOR+1) - 1];
                }
                if((data_index[i]/x) == 0){//first column
                    train_index[0 + tr * (NEIGHBOR+1)] = train_index[NEIGHBOR + tr * (NEIGHBOR+1)];
                }
                if((data_index[i]/x) == (y-1)){//last column
                    train_index[NEIGHBOR + tr * (NEIGHBOR+1)] = train_index[0 + tr * (NEIGHBOR+1)];
                }
            }
            if(NEIGHBOR == 8)
            {
                train_index[(NEIGHBOR/2) + tr * (NEIGHBOR+1) - 1] = data_index[i] - 1;
                train_index[(NEIGHBOR/2) + tr * (NEIGHBOR+1) + 1] = data_index[i] + 1;
                for(int j0=0;j0<3;j0++){
                    train_index[j0 + tr * (NEIGHBOR+1)] = data_index[i] - 1 - x + j0;
                    train_index[j0+6 + tr * (NEIGHBOR+1)] = data_index[i] - 1 + x + j0;
                }

                if((data_index[i] % x) == 0){//first row
                    for (int j=0; j<3; j++)
                        train_index[j*3 + tr*(NEIGHBOR+1)] = train_index[j*3+2 + tr*(NEIGHBOR+1)];
                }
                if((data_index[i] % x) == (x-1)){//last row
                    for(int j=0;j<3;j++)
                            train_index[j*3+2 + tr*(NEIGHBOR+1)] = train_index[j*3 + tr*(NEIGHBOR+1)];
                }
                if((data_index[i]/x) == 0){//first column
                    for(int j=0;j<3;j++)
                        train_index[j + tr*(NEIGHBOR+1)] = train_index[j+6 + tr*(NEIGHBOR+1)];
                }
                if((data_index[i]/x) == (y-1)){//last column
                    for(int j=0;j<3;j++)
                        train_index[j+6  + tr*(NEIGHBOR+1)] = train_index[j + tr*(NEIGHBOR+1)];
                }
            }

            int mid = int(labels[data_index[i]])-1 + tr*NEU_NUM2;
            processed_labels[mid] = 1;
            tr = tr + 1;
        }
        if(i%5 == 0){
            test_index[(NEIGHBOR/2) + te * (NEIGHBOR+1)] = data_index[i];//当前像素索引
            if(NEIGHBOR == 4)
            {
                test_index[(NEIGHBOR/2) + te * (NEIGHBOR+1) - 1] = data_index[i] - 1;
                test_index[(NEIGHBOR/2) + te * (NEIGHBOR+1) + 1] = data_index[i] + 1;
                test_index[0 + te * (NEIGHBOR+1)] = data_index[i] - x;
                test_index[NEIGHBOR+ te * (NEIGHBOR+1)] = data_index[i] + x;

                if((data_index[i] % x) == 0){//first row
                    test_index[(NEIGHBOR/2) + te * (NEIGHBOR+1) - 1] = test_index[(NEIGHBOR/2) + te * (NEIGHBOR+1) + 1];
                }
                if((data_index[i] % x) == (x-1)){//last row
                    test_index[(NEIGHBOR/2) + te * (NEIGHBOR+1) + 1] = test_index[(NEIGHBOR/2) + te * (NEIGHBOR+1) - 1];
                }
                if((data_index[i]/x) == 0){//first column
                    test_index[0 + te * (NEIGHBOR+1)] = test_index[NEIGHBOR+ te * (NEIGHBOR+1)];
                }
                if((data_index[i]/x) == (y-1)){//last column
                    test_index[NEIGHBOR+ te * (NEIGHBOR+1)] = test_index[0 + te * (NEIGHBOR+1)];
                }
            }
            if(NEIGHBOR == 8)
            {
                test_index[(NEIGHBOR/2) + te * (NEIGHBOR+1) - 1] = data_index[i] - 1;
                test_index[(NEIGHBOR/2) + te * (NEIGHBOR+1) + 1] = data_index[i] + 1;
                for(int j0=0;j0<3;j0++){
                    test_index[j0 + te * (NEIGHBOR+1)] = data_index[i] - 1 - x + j0;
                    test_index[j0+6 + te * (NEIGHBOR+1)] = data_index[i] - 1 + x + j0;
                }

                if((data_index[i] % x) == 0){//first row
                    for (int j=0; j<3; j++)
                        test_index[j*3 + te*(NEIGHBOR+1)] = test_index[j*3+2 + te*(NEIGHBOR+1)];
                }
                if((data_index[i] % x) == (x-1)){//last row
                    for(int j=0;j<3;j++)
                        test_index[j*3+2 + te*(NEIGHBOR+1)] = test_index[j*3 + te*(NEIGHBOR+1)];
                }
                if((data_index[i]/x) == 0){//first column
                    for(int j=0;j<3;j++)
                        test_index[j + te*(NEIGHBOR+1)] = test_index[j+6 + te*(NEIGHBOR+1)];
                }
                if((data_index[i]/x) == (y-1)){//last column
                    for(int j=0;j<3;j++)
                        test_index[j+6  + te*(NEIGHBOR+1)] = test_index[j + te*(NEIGHBOR+1)];
                }
            }
            test_labels[te] = labels[data_index[i]];
            te = te + 1;
        }
        
    }

    shuffle(train_index, processed_labels, (NEIGHBOR+1), train_size);
    
    checkCudaErrors(cudaMalloc((void **) &gpu_data, sizeof(double) * x * y * z));
    checkCudaErrors(cudaMemcpy(gpu_data, data, sizeof(double)* x * y * z, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **) &gpu_train_index, sizeof(int) * train_size * (NEIGHBOR+1)));
    checkCudaErrors(cudaMemcpy(gpu_train_index, train_index, sizeof(int) * train_size * (NEIGHBOR+1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &gpu_test_index, sizeof(int) * test_size * (NEIGHBOR+1)));
    checkCudaErrors(cudaMemcpy(gpu_test_index, test_index, sizeof(int) * test_size * (NEIGHBOR+1), cudaMemcpyHostToDevice));


    checkCudaErrors(cudaMalloc((void **) &gpu_processed_test, sizeof(double) * test_size * (NEIGHBOR+1) * z));
    checkCudaErrors(cudaMalloc((void **) &gpu_processed_train, sizeof(double) * train_size * (NEIGHBOR+1) * z));
    
    int gridsize = 64;
    int blocksize = 1024;
    
    double * processed_train = new double [train_size * (NEIGHBOR+1) * z];
    double * processed_test = new double [test_size * (NEIGHBOR+1) * z];
    
    int iter=0;
    
    preprocessing<<<gridsize,blocksize>>>(iter, gpu_data, gpu_train_index, gpu_processed_train, x, y, z, train_size);
    preprocessing<<<gridsize,blocksize>>>(iter, gpu_data, gpu_test_index, gpu_processed_test, x, y, z, test_size);

    cudaDeviceSynchronize();
    end = clock();
    double tt = double(end - start);
    fprintf(stdout,"[Samples prepared with %d Nearest-Neighbor-Pixels Strategy] Proportion of Training samples: %d%%  Execution time: %.3f sec\n", NEIGHBOR, 80, tt/CLOCKS_PER_SEC);

    checkCudaErrors(cudaFree(gpu_data));
    checkCudaErrors(cudaFree(gpu_train_index));
    checkCudaErrors(cudaFree(gpu_test_index));
    cudaDeviceSynchronize();

    
    int re_size = 0;
    for ( int i = 0; i + COV_LEN < z; i += STRIDE){
        re_size ++;
    }

    int mre_size = (re_size - 1) / POOLING_LEN + 1;
    int pooling_input_length = re_size * FILTERS_NUM;
    int pooling_output_length = mre_size * FILTERS_NUM;
    int ful_weights_size = mre_size * FILTERS_NUM * NEU_NUM1;
    int out_weights_size = NEU_NUM1 * NEU_NUM2;
    int filter_size = (NEIGHBOR + 1) * COV_LEN;
    int cube_size = (NEIGHBOR + 1) * z;

    double * gpu_loss;
    
    checkCudaErrors(cudaMalloc((void**) &gpu_processed_labels, sizeof(double) * train_size * NEU_NUM2));
    checkCudaErrors(cudaMemcpy(gpu_processed_labels,processed_labels,sizeof(double) * train_size * NEU_NUM2,cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **) &gpu_loss, sizeof(double) * 2));
    
    double loss;
    double * logloss = new double [300]();
    double * correct_rate = new double [VALID_BATCH];
    for(int i=0; i<VALID_BATCH; i++){
        correct_rate[i] = 1;
    }

    double cur_min = 1;
    int count = 1;
    bool tag = true; // for training
    int max_iter = 300;
    start = clock();

    Layer conv(cube_size, filter_size * FILTERS_NUM, FILTERS_NUM, pooling_input_length, false);
    Layer maxpol(pooling_input_length, pooling_input_length, pooling_output_length, pooling_output_length, false);
    Layer fulconnect(pooling_output_length, pooling_output_length * NEU_NUM1, NEU_NUM1, NEU_NUM1, false);
    Layer outp(NEU_NUM1, NEU_NUM1 * NEU_NUM2, NEU_NUM2, NEU_NUM2, true);

    clock_t iter_start, iter_stop;
    fprintf(stdout, "[Cube CNN training with SGD algo] lr = %lf\n", learning_rate);
    for ( int j = 0; j < max_iter; j ++)
    {
        loss = 0;
        iter_start = clock();
        for ( int epoch_id = 0; epoch_id < train_size; epoch_id ++ )
        {
            checkCudaErrors(cudaMemset(maxpol.deltaW.data_d, 0, sizeof(double) * pooling_input_length));

            convolution<<< FILTERS_NUM, re_size, (cube_size + filter_size) * sizeof(double) >>>( epoch_id, 
                                                                                                 (NEIGHBOR + 1),
                                                                                                 z,
                                                                                                 COV_LEN,
                                                                                                 FILTERS_NUM,
                                                                                                 STRIDE,
                                                                                                 gpu_processed_train,
                                                                                                 conv.weights.data_d, 
                                                                                                 conv.bias.data_d,
                                                                                                 conv.output.data_d );

            maxpooling<<< FILTERS_NUM, mre_size >>>( re_size,
                                                     POOLING_LEN,
                                                     FILTERS_NUM,
                                                     conv.output.data_d, 
                                                     maxpol.output.data_d,
                                                     maxpol.bias.data_d );

            fully_connect<<< NEU_NUM1, pooling_output_length, pooling_output_length * sizeof(double) >>>( pooling_output_length,
                                                                                                          NEU_NUM1,
                                                                                                          maxpol.output.data_d, 
                                                                                                          fulconnect.weights.data_d, 
                                                                                                          fulconnect.bias.data_d, 
                                                                                                          fulconnect.output.data_d );
            
            output_and_loss<<< 1, NEU_NUM2, (NEU_NUM1 + 2 * NEU_NUM2) * sizeof(double) >>>( tag, 
                                                                                            epoch_id,
                                                                                            NEU_NUM1,
                                                                                            NEU_NUM2, 
                                                                                            fulconnect.output.data_d, 
                                                                                            outp.weights.data_d, 
                                                                                            outp.bias.data_d, 
                                                                                            outp.output.data_d,
                                                                                            gpu_processed_labels,
                                                                                            outp.deltaB.data_d,
                                                                                            gpu_loss );

            // backward propagation
            bp_fully_connect<<< NEU_NUM1, NEU_NUM2, NEU_NUM2 * sizeof(double) >>>( NEU_NUM1,
                                                                                   NEU_NUM2,
                                                                                   learning_rate, 
                                                                                   outp.weights.data_d, 
                                                                                   outp.bias.data_d, 
                                                                                   fulconnect.output.data_d, 
                                                                                   outp.deltaB.data_d, 
                                                                                   fulconnect.deltaB.data_d );
            
            bp_fully_connect<<< pooling_output_length, NEU_NUM1, NEU_NUM1 * sizeof(double) >>>( pooling_output_length, 
                                                                                                NEU_NUM1,
                                                                                                learning_rate,
                                                                                                fulconnect.weights.data_d, 
                                                                                                fulconnect.bias.data_d, 
                                                                                                maxpol.output.data_d, 
                                                                                                fulconnect.deltaB.data_d, 
                                                                                                maxpol.deltaB.data_d);

            bp_maxpooling<<< 1, pooling_output_length >>>( pooling_output_length,
                                                           maxpol.bias.data_d,
                                                           maxpol.deltaB.data_d,
                                                           maxpol.deltaW.data_d );
            
            bp_convolution<<< FILTERS_NUM, (NEIGHBOR+1)*COV_LEN, (NEIGHBOR+1) * z * sizeof(double)>>>(epoch_id, 
                                                                                                      STRIDE,
                                                                                                      (NEIGHBOR + 1),
                                                                                                      z,
                                                                                                      (NEIGHBOR + 1) * COV_LEN,
                                                                                                      FILTERS_NUM,
                                                                                                      pooling_input_length,
                                                                                                      learning_rate,
                                                                                                      maxpol.deltaW.data_d,
                                                                                                      conv.weights.data_d,
                                                                                                      conv.bias.data_d, 
                                                                                                      gpu_processed_train, 
                                                                                                      conv.output.data_d );

        }

        checkCudaErrors(cudaMemcpy(&loss, gpu_loss, sizeof(double), cudaMemcpyDeviceToHost));
        double single_rate = loss/train_size;
            logloss[j] = single_rate;
        if(single_rate < MIN_ERR)
            break;

        iter_stop = clock();
        float iter_time = float(iter_stop - iter_start) / CLOCKS_PER_SEC;
        char str[50];
        sprintf(str, "%d", j + 1);
        strcat(str, ",");
        fprintf(stdout,"[Cube CNN training with SGD algo. Execution time: %.3f sec] Iteration %-4s loss = %lf;\n", iter_time, str, single_rate);

        insert_line(correct_rate,single_rate);
        double new_min = *min_element(correct_rate, correct_rate + VALID_BATCH);
            if(cur_min > new_min){
                    cur_min = new_min;
                 count = 1;
            }
            else{
                    count++;
            }
            if(count >= VALID_BATCH) {
                    learning_rate = learning_rate * 0.9;
                    fprintf(stdout,"[Cube CNN training with SGD algo] lr = %lf\n", learning_rate);
                    count = 1;
                    cur_min = new_min;
            }        
    }

    fprintf(stdout, "[Cube CNN training with SGD algo] ");
    end = clock();
    tt = double(end - start);
    fprintf(stdout,"Completed! Global Execution time is %.3f sec\n", tt/CLOCKS_PER_SEC);

    // start = clock();
    checkCudaErrors(cudaMemcpy(conv.weights.data_h, conv.weights.data_d, sizeof(double) * (NEIGHBOR+1) * COV_LEN * FILTERS_NUM, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(conv.bias.data_h, conv.bias.data_d, sizeof(double) * FILTERS_NUM, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(fulconnect.bias.data_h, fulconnect.bias.data_d, sizeof(double) * NEU_NUM1, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(outp.bias.data_h, outp.bias.data_d, sizeof(double) * NEU_NUM2, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(fulconnect.weights.data_h, fulconnect.weights.data_d, sizeof(double) * ful_weights_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(outp.weights.data_h, outp.weights.data_d, sizeof(double) * out_weights_size, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    MATFile * pmatFile;
    pmatFile = matOpen("model.mat","w");
    mxArray * m1 = mxCreateDoubleMatrix((NEIGHBOR+1)*COV_LEN, FILTERS_NUM,mxREAL);
    memcpy((void *)mxGetPr(m1), (void *)conv.weights.data_h, sizeof(double) * (NEIGHBOR+1) * COV_LEN * FILTERS_NUM);
    matPutVariable(pmatFile, "filters", m1);

    mxArray * m2 = mxCreateDoubleMatrix(FILTERS_NUM,1,mxREAL);
    memcpy((void *)mxGetPr(m2), (void *)conv.bias.data_h, sizeof(double) * FILTERS_NUM);
    matPutVariable(pmatFile, "conv_bias", m2);

    mxArray * m3 = mxCreateDoubleMatrix(NEU_NUM1,pooling_output_length,mxREAL);
    memcpy((void *)mxGetPr(m3), (void *)fulconnect.weights.data_h, sizeof(double) * ful_weights_size);
    matPutVariable(pmatFile, "fulconnect_weights", m3);

    mxArray * m4 = mxCreateDoubleMatrix(NEU_NUM1,1,mxREAL);
    memcpy((void *)mxGetPr(m4), (void *)fulconnect.bias.data_h, sizeof(double) * NEU_NUM1);
    matPutVariable(pmatFile, "fulconnect_bias", m4);

    mxArray * m5 = mxCreateDoubleMatrix(NEU_NUM2,NEU_NUM1,mxREAL);
    memcpy((void *)mxGetPr(m5), (void *)outp.weights.data_h, sizeof(double) * out_weights_size);
    matPutVariable(pmatFile, "output_weights", m5);

    mxArray * m6 = mxCreateDoubleMatrix(NEU_NUM2,1,mxREAL);
    memcpy((void *)mxGetPr(m6), (void *)outp.bias.data_h, sizeof(double) * NEU_NUM2);
    matPutVariable(pmatFile, "output_bias", m6);

    mxArray * m7 = mxCreateDoubleMatrix(300,1,mxREAL);
    memcpy((void *)mxGetPr(m7), (void *)logloss, sizeof(double) * 300);
    matPutVariable(pmatFile, "loss", m7);

    matClose(pmatFile);
    
    // test
    start = clock();
    double right = 0;
    double count0 = 0;
    tag = false; // for testing
    for ( int i1 = 0; i1 < test_size; i1 ++ ) {
        convolution<<< FILTERS_NUM,re_size, (cube_size + filter_size) * sizeof(double) >>>( i1,
                                                                                            (NEIGHBOR + 1),
                                                                                            z,
                                                                                            COV_LEN,
                                                                                            FILTERS_NUM,
                                                                                            STRIDE,
                                                                                            gpu_processed_test,
                                                                                            conv.weights.data_d,
                                                                                            conv.bias.data_d, 
                                                                                            conv.output.data_d );
        
        maxpooling<<< FILTERS_NUM, mre_size >>>( re_size,
                                                 POOLING_LEN,
                                                 FILTERS_NUM,
                                                 conv.output.data_d, 
                                                 maxpol.output.data_d, 
                                                 maxpol.bias.data_d );
        
        fully_connect<<< NEU_NUM1, pooling_output_length, pooling_output_length * sizeof(double) >>>( pooling_output_length,
                                                                                                      NEU_NUM1,
                                                                                                      maxpol.output.data_d, 
                                                                                                      fulconnect.weights.data_d,
                                                                                                      fulconnect.bias.data_d,
                                                                                                      fulconnect.output.data_d );
        
        output_and_loss<<< 1, NEU_NUM2, (NEU_NUM1 + 2 * NEU_NUM2) *sizeof(double) >>>( tag, 
                                                                                       i1,
                                                                                       NEU_NUM1,
                                                                                       NEU_NUM2, 
                                                                                       fulconnect.output.data_d, 
                                                                                       outp.weights.data_d, 
                                                                                       outp.bias.data_d, 
                                                                                       outp.output.data_d,
                                                                                       NULL,
                                                                                       NULL,
                                                                                       NULL ); 
        
        checkCudaErrors(cudaMemcpy(outp.output.data_h, outp.output.data_d, sizeof(double) * NEU_NUM2, cudaMemcpyDeviceToHost));
        
        right = count_err(test_labels, outp.output.data_h, i1);
        count0 = count0 + right;
    }
    end = clock();
    tt = double(end - start);
    fprintf(stdout,"[Cube CNN testing] Execution time is %.3f sec. ", tt/CLOCKS_PER_SEC);
    return count0/test_size;
}

int main(int argc, char * argv[])
{
    fprintf(stdout, "[Cube CNN training with SGD algo] ");
      if(!InitCUDA()){
        return 0;
    }
    printf("CUDA initialized.\n");

    fprintf(stdout, "[Cube CNN training with SGD algo] Available Device List: ");
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    int device = 0;
    for ( device = 0; device < deviceCount; device ++ )
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        if ( device == 0 )
            printf("Device %d -- %s(Default)  ", device, deviceProp.name);
        else
            printf("Device %d -- %s  ", device, deviceProp.name);
    }
    cout << endl;

    double *trainset,*trainlabels;
    if ( argc != 3 ) {
        fprintf(stderr, "3 input arguments required!");
    }

    MATFile * datamat = matOpen(argv[1], "r");
    mxArray * train = matGetVariable(datamat,"DataSet");
    mxArray * labels = matGetVariable(datamat,"labels");
    
    int device_choosed = (int)atoi(argv[2]);
    fprintf(stdout, "[Cube CNN training with SGD algo] Training implemented on Device %d.\n", device_choosed);

    checkCudaErrors(cudaSetDevice(device_choosed));

    trainset = (double*)mxGetData(train);
    trainlabels = (double*)mxGetData(labels);
    
    const mwSize  * dim;
    dim = mxGetDimensions(train);

    double correct = training(trainset, trainlabels, dim[0], dim[1], dim[2]);
    fprintf(stdout,"Accuracy: %.3f%% \n", correct * 100);
    
    cudaDeviceReset();
    return 0;
}
