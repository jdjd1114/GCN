#include <mat.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <matrix.h>
#include <iostream>
#include "error_util.h"
#include <cuda_runtime.h>
#include <algorithm>
using namespace std;

const int FILTER_NUM = 20;//number of filters
const int P_NUM = 19;//number of layers of each convolution operation
const int stride = 2;
const int GP_NUM = 2;//size of each group
const int NEU_NUM1 = 100;//number of neurons in full connection layer
const int NEU_NUM2 = 13;//output layer
const int NEIGHBOR = 8;//number of neighbor pixels
double LEARN_RATE = 0.2;
const double MIN_ERR = 0.001;
const int VALID_BATCH = 5;
const int DATA_BATCH = 100;//batch size

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
		fprintf(stderr,"There is no device supporting CUDA.\n");
		return false;
	}
	cudaSetDevice(i);
	return true;
}

struct Tensor{
    int length;
    double * data_h;
    double * data_d;

    Tensor();
};

Tensor::Tensor()
{
    length = 0;
    data_h = NULL;
    data_d = NULL;
}

struct DataLayer{
    Tensor input;
    Tensor labels;

    DataLayer(int input_size, int labels_size);
    ~DataLayer();
};

DataLayer::DataLayer(int input_size, int labels_size)
{
    input.length = input_size;
    labels.length = labels_size;
}

DataLayer::~DataLayer()
{
    if(input.data_h != NULL)
        delete [] input.data_h;
    if(labels.data_h != NULL)
        delete [] labels.data_h;
    if(input.data_d != NULL)
        cudaFree(input.data_d);
    if(labels.data_d != NULL)
        cudaFree(labels.data_d);
}

struct Layer{
    Tensor input;
    Tensor output;
    Tensor weights;
    Tensor bias;
    Tensor deltaW;
    Tensor deltaB;

    Layer(int input_size, int weights_size, int bias_size, int output_size, int batch_size, bool copyback);
    ~Layer();

private:
    void allocMemcpyCuda(int size, double ** data_h, double ** data_d, bool isMalloc, bool isCopyback);
};

void Layer::allocMemcpyCuda(int size, double **data_h, double **data_d, bool isMallochost, bool isInitalize)
{
    size_t sizeBytes = size * sizeof(double);
    checkCudaErrors(cudaMalloc(data_d, sizeBytes));

    if (isMallochost)
    {
        *data_h = new double [size];

        if (isInitalize)
        {
            for (int i = 0; i < size; i ++)
            {
                data_h[0][i] = (2 * (rand()/double(RAND_MAX)) - 1) / 50;
            }
            checkCudaErrors(cudaMemcpy(*data_d, *data_h, sizeBytes, cudaMemcpyHostToDevice));
        }
    }
}

Layer::Layer (int input_size, int weights_size, int bias_size, int output_size, int batch_size, bool copyback)
{
    input.length = input_size * batch_size;
    weights.length = weights_size;
    deltaW.length = weights_size * batch_size;
    output.length = output_size * batch_size;
    bias.length = bias_size;
    deltaB.length = bias_size * batch_size;

    allocMemcpyCuda(input.length, &input.data_h, &input.data_d, false, false);
    allocMemcpyCuda(weights.length, &weights.data_h, &weights.data_d, true, true);
    allocMemcpyCuda(bias.length, &bias.data_h, &bias.data_d, true, true);
    allocMemcpyCuda(output.length, &output.data_h, &output.data_d, copyback, false);
    allocMemcpyCuda(deltaB.length, &deltaB.data_h, &deltaB.data_d, false, false);
    allocMemcpyCuda(deltaW.length, &deltaW.data_h, &deltaW.data_d, false, false);
}

Layer::~Layer ()
{
    if (input.data_h != NULL)
        delete [] input.data_h;
    if(weights.data_h != NULL)
        delete [] weights.data_h;
    if(output.data_h != NULL)
        delete [] output.data_h;
    if(bias.data_h != NULL)
        delete [] bias.data_h;
    if(input.data_d != NULL)
        cudaFree(input.data_d);
    if(output.data_d != NULL)
        cudaFree(output.data_d);
    if(weights.data_d != NULL)
        cudaFree(weights.data_d);
    if(bias.data_d != NULL)
        cudaFree(bias.data_d);
    if(deltaW.data_d != NULL)
        cudaFree(deltaW.data_d);
    if(deltaB.data_d != NULL)
        cudaFree(deltaB.data_d);
}

//copy data to shared memory
__device__ void copy_data_to_shared(double * data, double * data_tmp, int tid, int offset, int head, int length)
{
	for(size_t i = tid * offset; i < (tid + 1) * offset && (i < length); i++)
    {
		data_tmp[i] = data[i + head];
	}
	__syncthreads();

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// forward propagation kernels
// forward convolution
__global__ static void convolution( int data_id,
                                    int batch_id,
                                    int z,
                                    int stride,
                                    int output_size,
                                    double * input, 
                                    double * filters, 
                                    double * bias,
                                    double * output )
{

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	if(tid < output_size && bid < FILTER_NUM)
	{
		extern __shared__ double input_tmp[];
        int head = data_id * (NEIGHBOR + 1) * z;
		int length = (NEIGHBOR + 1) * z;
        int offset = (length - 1)/output_size + 1;
		copy_data_to_shared(input, input_tmp, tid, offset, head, length);
        
		__shared__ double filters_tmp[ (NEIGHBOR + 1) * P_NUM ];
		head = bid * (NEIGHBOR + 1) * P_NUM;
		length = (NEIGHBOR + 1) * P_NUM;
		offset = (length - 1)/output_size + 1;
		copy_data_to_shared(filters, filters_tmp, tid, offset, head, length);
		__syncthreads();

		double mid = 0;
		for(int i = 0; i < (NEIGHBOR + 1) * P_NUM; i++){
            mid = mid + filters_tmp[i] * input_tmp[tid * (NEIGHBOR + 1) * stride + i];
		}
		mid = mid + bias[bid];

        output[tid + bid * output_size + batch_id * output_size * FILTER_NUM] = 2/(1 + (1/exp(2*mid))) - 1;
	}
}

// forward maxpooling
__global__ static void maxpooling( int batch_id,
                                   int input_size,
                                   int output_size,
                                   double * input,
                                   double * output, 
                                   double * output_index )
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid < output_size && bid < FILTER_NUM){
		double max;
		int max_index, head, tail;
		head = tid * GP_NUM + bid * input_size + batch_id * input_size * FILTER_NUM;
		tail = (tid+1) * GP_NUM + bid * input_size + batch_id * input_size * FILTER_NUM;
		max = input[head];
		max_index = head;
		for(int i=head; i < tail && (i < (bid + 1) * input_size + batch_id * input_size * FILTER_NUM); i++)
        {
			if(max < input[i]){
				max = input[i];
				max_index=i;
			}
		}

		output[tid + bid * output_size + batch_id * output_size * FILTER_NUM] = max;
		output_index[tid + bid * output_size + batch_id * output_size * FILTER_NUM] = max_index;
	}
}

// forward fully connection
__global__ static void fully_connect( int batch_id,
                                      int mre_size,
                                      double * mre,
                                      double * omega,
                                      double * bias,
                                      double * F1 )
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if( tid < (mre_size * FILTER_NUM) && bid < NEU_NUM1 )
    {
		extern __shared__ double ner[];
		ner[tid] = omega[bid + tid*NEU_NUM1] * mre[tid + batch_id * mre_size * FILTER_NUM];
		__syncthreads();//waiting for other threads

		int length = mre_size * FILTER_NUM;
		int offset = (length - 1)/2 + 1;

		while(length >= 2)
		{
			if( tid + offset < length )
			{
				ner[tid] = ner[tid] + ner[tid + offset];
			}
			offset = (offset - 1)/2 + 1;
			length = (length - 1)/2 + 1;
			__syncthreads();
		}

		F1[bid + batch_id * NEU_NUM1] = 2/(1 + 1/exp((ner[0] + bias[bid]) * 2)) - 1;
	}
}

// forward output
__global__ static void output( int batch_id, 
                               double * F1, 
                               double * omega2, 
                               double * bias, 
                               double * O2)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < NEU_NUM2){
		//copy F1 to shared memory
		__shared__ double F1_tmp[NEU_NUM1];
		int offset = (NEU_NUM1 - 1) / NEU_NUM2 + 1;
		copy_data_to_shared(F1, F1_tmp, id, offset, batch_id*NEU_NUM1, NEU_NUM1);
		__syncthreads();
		__shared__ double O2_tmp[1][NEU_NUM2];

		//compute the output of a neuron
		double mid = 0;
		for(int i=0; i<NEU_NUM1; i++){
			mid = mid + omega2[id + i*NEU_NUM2] * F1_tmp[i];
		}
		O2[id + batch_id*NEU_NUM2] = exp(mid+ bias[id]);
		O2_tmp[0][id] = O2[id + batch_id*NEU_NUM2];
		__syncthreads(); //waiting for other threads

		int length = NEU_NUM2;//length of the array needed to be summed up
		offset = (length - 1)/2 +1;//bias value
		while(length >= 2)
		{
			if(id + offset < length){
				O2_tmp[0][id] = O2_tmp[0][id] + O2_tmp[0][id + offset];
			}
			offset = (offset - 1)/2 + 1;
			length = (length - 1)/2 + 1;
			__syncthreads();//waiting for all threads
		}
		O2[id + batch_id*NEU_NUM2] = O2[id + batch_id*NEU_NUM2]/O2_tmp[0][0];

	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// backward propagation kernels
// output layer
__global__ static void bp_output( int iter, 
                                  int train_idx, 
                                  int batch_id, 
                                  double LEARN_RATE, 
                                  double * labels, 
                                  double * O2, 
                                  double * delta_L_z )
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;
	int id = tid + iter * threadNum;
	if(id < NEU_NUM2){
		delta_L_z[id + batch_id*NEU_NUM2] = (O2[id + batch_id*NEU_NUM2] - labels[id + train_idx * NEU_NUM2])/NEU_NUM2;
	}
}

// fully_connect
__global__ static void bp_fully_connect( int batch_id, 
                                         double LEARN_RATE, 
                                         double * omega2, 
                                         double * delta_L_z, 
                                         double * delta_f_w,
                                         double * F1, 
                                         double * delta_f_z )
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid < NEU_NUM2 && bid < NEU_NUM1){
		__shared__ double dfa[1][NEU_NUM2];
		dfa[0][tid] = omega2[tid + bid*NEU_NUM2] * delta_L_z[tid + batch_id*NEU_NUM2];
		__syncthreads();

		delta_f_w[tid + bid*NEU_NUM2 + batch_id*NEU_NUM1*NEU_NUM2] = F1[bid + batch_id*NEU_NUM1] * delta_L_z[tid + batch_id*NEU_NUM2]; 

		int length = NEU_NUM2;
		int offset = (length - 1)/2 + 1;
		while(length >= 2){
			if(tid + offset < length){
				dfa[0][tid] = dfa[0][tid] + dfa[0][tid+offset];
			}
			length = (length - 1)/2 + 1;
			offset = (offset - 1)/2 + 1;
			__syncthreads();
		}

		delta_f_z[bid + batch_id*NEU_NUM1] = dfa[0][0] * (1 + F1[bid +batch_id*NEU_NUM1]) * (1 - F1[bid + batch_id*NEU_NUM1]);
	}
}

// maxpooling layer
__global__ static void bp_maxpooling(int batch_id, 
                                     double LEARN_RATE, 
                                     int re_size, 
                                     int mre_size, 
                                     double * mre,
                                     double * mre_index,
                                     double * delta_22,
                                     double * omega1,
                                     double * delta_f_z,
                                     double * delta_m_w)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if( tid < NEU_NUM1 && bid < mre_size * FILTER_NUM )
    {
		__shared__ double mid[1][NEU_NUM1];
		mid[0][tid] = omega1[tid + bid*NEU_NUM1] * delta_f_z[tid + batch_id*NEU_NUM1];
		__syncthreads();//waiting for all threads
		int length = NEU_NUM1;
		int offset = (length - 1)/2 + 1;
		while(length >= 2){
			if(tid + offset < length){
				mid[0][tid] = mid[0][tid] + mid[0][tid+offset];
			}
			length = (length - 1)/2 + 1;
			offset = (offset - 1)/2 + 1;
			__syncthreads();
		}

		delta_m_w[tid + bid*NEU_NUM1 + batch_id * mre_size * FILTER_NUM * NEU_NUM1] = mre[bid + batch_id * mre_size * FILTER_NUM] * 
                                                                                      delta_f_z[tid + batch_id * NEU_NUM1];

        //int idx = (int)(mre_index[bid + batch_id * mre_size * FILTER_NUM]) + batch_id * re_size * FILTER_NUM;
		if(tid < 1)
			delta_22[ (int)(mre_index[bid + batch_id * mre_size * FILTER_NUM]) + 
                      batch_id * re_size * FILTER_NUM] = mid[0][0] * 
                                                         (1 + mre[bid + batch_id * mre_size * FILTER_NUM]) * 
                                                         (1 - mre[bid + batch_id * mre_size * FILTER_NUM]);
	}
}

// convolutional layer
__global__ static void bp_convolution( int data_id, 
                                       int batch_id, 
                                       double LEARN_RATE, 
                                       int z, 
                                       int stride, 
                                       int mre_size,
                                       int re_size, 
                                       double * mre_index, 
                                       double * deltaZ, 
                                       double * deltaW, 
                                       double * deltaB, 
                                       double * data )
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid < ((NEIGHBOR+1)*P_NUM) && (bid < FILTER_NUM))
	{
		int head = data_id * (NEIGHBOR + 1) * z;
		int length = (NEIGHBOR + 1) * z;
		int offset = (length - 1)/((NEIGHBOR + 1)* P_NUM) + 1;
		extern __shared__ double train_tmp[];
		copy_data_to_shared(data, train_tmp, tid, offset, head, length);
        __syncthreads();

        __shared__ double delta_W[1][(NEIGHBOR+1)*P_NUM];
		delta_W[0][tid] = 0;

		int idx, n, i, h;
		double mid = 0;
		for(i = 0; i < mre_size; i++){
			idx = int(mre_index[i + bid * mre_size + batch_id * mre_size * FILTER_NUM]);
			n = idx % re_size;
			h = n * (NEIGHBOR + 1) * stride;
			delta_W[0][tid] = delta_W[0][tid] + deltaZ[idx + batch_id * re_size * FILTER_NUM] * train_tmp[tid + h];
			mid = mid + deltaZ[idx + batch_id * re_size * FILTER_NUM];
		}

		deltaW[tid + bid * (NEIGHBOR + 1) * P_NUM + batch_id * (NEIGHBOR + 1) * P_NUM * FILTER_NUM] = delta_W[0][tid] / mre_size;
		
		if(tid < 1)
			deltaB[bid + batch_id*FILTER_NUM] = (mid/mre_size);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// update params kernels
// output layer
__global__ static void update_output( int batch_size, 
                                      double LEARN_RATE, 
                                      double * delta_L_z, 
                                      double * bias2 )
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid < batch_size && bid < NEU_NUM2)
    {
		__shared__ double mid[1][DATA_BATCH];
		mid[0][tid] = delta_L_z[bid + tid*NEU_NUM2];
		__syncthreads();//waiting for all threads

		int length = batch_size;
		int offset = (length - 1)/2 + 1;
		while(length >= 2){
			if(tid + offset < length){
				mid[0][tid] = mid[0][tid] + mid[0][tid + offset];
			}
			length = (length - 1)/2 + 1;
			offset = (offset - 1)/2 + 1;
			__syncthreads();
		}

		if(tid < 1)
			bias2[bid] = bias2[bid] - mid[0][0] * LEARN_RATE / batch_size;
	}
}
// fully_connect layer
__global__ static void update_fully_connect(int batch_size, double LEARN_RATE, double * omega2,double * bias1,double *delta_f_w, double * delta_f_z)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid<NEU_NUM2 && bid<NEU_NUM1){
		double mid0 = 0, mid1 = 0;
		for(int i=0; i<batch_size; i++){
			mid0 = mid0 + delta_f_w[tid + bid*NEU_NUM2 + i*NEU_NUM1*NEU_NUM2];
			mid1 = mid1 + delta_f_z[bid + i*NEU_NUM1];
		}
		omega2[tid + bid*NEU_NUM2] = omega2[tid + bid] - LEARN_RATE * mid0/batch_size;
		
		if(tid < 1){
			bias1[bid] = bias1[bid] - LEARN_RATE * mid1 / batch_size;
		}
	}
}
// maxpooling layer
__global__ static void update_maxpooling( int mre_size, 
                                          int batch_size, 
                                          double LEARN_RATE, 
                                          double * omega1, 
                                          double * delta_m_w )
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid<NEU_NUM1 && bid<mre_size * FILTER_NUM){
		double mid=0;
		for(int i=0; i<batch_size; i++){
			mid = mid + delta_m_w[tid + bid * NEU_NUM1 + i * mre_size * FILTER_NUM * NEU_NUM1];
		}
		
		omega1[tid + bid*NEU_NUM1] = omega1[tid + bid*NEU_NUM1] - LEARN_RATE*mid/batch_size;
	}
}

// convolution layer
__global__ static void update_convolution(int batch_size, int re_size, double LEARN_RATE, double * delta_22, double * delta_k_w, double * delta_k_b, double * filters, double * bias0)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid < ((NEIGHBOR+1)*P_NUM) && (bid < FILTER_NUM))
	{
		double mid0 = 0, mid1 = 0;
		for(int i=0; i<batch_size; i++){
			mid0 = mid0 + delta_k_w[tid + bid*(NEIGHBOR+1)*P_NUM +i*(NEIGHBOR+1)*P_NUM*FILTER_NUM];
			mid1 = mid1 + delta_k_b[bid + i*FILTER_NUM];
		}
		filters[tid + bid*(NEIGHBOR+1)*P_NUM] = filters[tid + bid*(NEIGHBOR+1)*P_NUM] - LEARN_RATE*mid0/batch_size;
		
		if(tid < 1){
			bias0[bid] = bias0[bid] - LEARN_RATE*mid1/batch_size;
		}	
	}
}

__global__ static void loss_function(int batch_id, int batch_size, double * O2, double * labels, double * loss_values)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    double tmp = 0.0;
    if(tid < batch_size){
        for(size_t i=0; i<NEU_NUM2; i++){
            tmp = tmp + labels[i + (batch_id * DATA_BATCH + tid) * NEU_NUM2] * log(O2[i + tid * NEU_NUM2]) +
                  (1 - labels[i + (batch_id * DATA_BATCH + tid) * NEU_NUM2]) * log(1 - O2[i + tid * NEU_NUM2]);
        }

        loss_values[tid] = -tmp/NEU_NUM2;
    }
}

//preprocessing
__global__ static void processing(int iter, double * data, int * train_index, double * processed_data, int x, int y, int z, int train_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;
	int id = tid + iter * threadNum;

	//int idx = id * (NEIGHBOR+1) * z;
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
        //(output[i] - labels[i + idx*NEU_NUM2]) * (output[i] - labels[i + idx*NEU_NUM2]);
	}
	l = -l/NEU_NUM2;
	return l;
}


//compute correct rate
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

//Insert current loss value to the queue
void insert_line(double * a, double b){
	for(int i=1; i<VALID_BATCH; i++){
		a[i-1] = a[i];
	}
	a[VALID_BATCH-1] = b;
}

//shuffle
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

//training
double training(double * data, double * labels, int x, int y, int z){
	clock_t start, end;
	start = clock();	
	double * gpu_data;//original hyperspectral image, saved in global memory
	double * gpu_processed_train;//extracted train samples
	double * gpu_processed_test;//extracted test samples
	int * gpu_train_index;//index of train samples and their neighbors
	int * gpu_test_index;//index of test samples
	double * gpu_processed_labels;//encoded train labels

	//preprocessing
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
	//fprintf(stdout,"train_size:%d  test_size:%d\n",train_size,test_size);
	int * train_index = new int [train_size * (NEIGHBOR + 1)];
	int * test_index = new int [test_size * (NEIGHBOR+1)];

	double * processed_labels = new double [train_size * NEU_NUM2]();
	double * test_labels = new double [test_size]();

	int tr=0, te=0;
	for (int i=0; i<data_size; i++){
		if (i%5 != 0){
			train_index[(NEIGHBOR/2) + tr * (NEIGHBOR+1)] = data_index[i];//index of current labeled pixel
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
			test_index[(NEIGHBOR/2) + te * (NEIGHBOR+1)] = data_index[i];//index of current labeled pixel
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

			//int mid = int(labels[data_index[i]])-1 + te*NEU_NUM2;
			test_labels[te] = labels[data_index[i]];
			te = te + 1;
		}
	}

	shuffle(train_index, processed_labels, (NEIGHBOR+1), train_size);//shuffle the samples in training set

	//malloc GPU memory, copy data to GPU
	checkCudaErrors(cudaMalloc((void **) &gpu_data, sizeof(double) * x * y * z));
	checkCudaErrors(cudaMemcpy(gpu_data, data, sizeof(double)* x * y * z, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &gpu_train_index, sizeof(int) * train_size * (NEIGHBOR+1)));
	checkCudaErrors(cudaMemcpy(gpu_train_index, train_index, sizeof(int) * train_size * (NEIGHBOR+1), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void **) &gpu_test_index, sizeof(int) * test_size * (NEIGHBOR+1)));
	checkCudaErrors(cudaMemcpy(gpu_test_index, test_index, sizeof(int) * test_size * (NEIGHBOR+1), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &gpu_processed_test, sizeof(double) * test_size * (NEIGHBOR+1) * z));
    checkCudaErrors(cudaMalloc((void **) &gpu_processed_train, sizeof(double) * train_size * (NEIGHBOR+1) *z));

    delete [] data_index;
    delete [] train_index;
    delete [] test_index;

    int gridsize = 64;
    int blocksize = 512;
	int iter=0;

	processing<<<gridsize,blocksize>>>(iter, gpu_data, gpu_train_index, gpu_processed_train, x, y, z, train_size);
	processing<<<gridsize,blocksize>>>(iter, gpu_data, gpu_test_index, gpu_processed_test, x, y, z, test_size);

	//cudaDeviceSynchronize();
	end = clock();
	float tt = float(end - start);
	fprintf(stdout,"[Samples prepared with %d Nearest-Neighbor-Pixels Strategy  Proportion of Training Samples : %d%%] Execution time : %.3f sec\n", 
            NEIGHBOR, 80, tt/CLOCKS_PER_SEC);

	checkCudaErrors(cudaFree(gpu_data));
	checkCudaErrors(cudaFree(gpu_train_index));
	checkCudaErrors(cudaFree(gpu_test_index));
	
	// calculate length of convolutional results
	int re_size = 0;
	for ( int i = 0; i + P_NUM - 1 < z; i += stride )
    {
		re_size ++;
	}
	int mre_size = (re_size-1)/GP_NUM + 1;
	int ome_num1 = mre_size * FILTER_NUM * NEU_NUM1;// Weights in full connection layer
	int ome_num2 = NEU_NUM1 * NEU_NUM2;// Weights in output layer
	
    double * gpu_loss_values;
    double * gpu_out_deltaW;

	// copy labels to GPU
	checkCudaErrors(cudaMalloc((void**) &gpu_processed_labels, sizeof(double) * train_size * NEU_NUM2));
	checkCudaErrors(cudaMemcpy(gpu_processed_labels,processed_labels,sizeof(double) * train_size * NEU_NUM2,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &gpu_loss_values, sizeof(double) * DATA_BATCH));
    checkCudaErrors(cudaMalloc((void **) &gpu_out_deltaW, sizeof(double) * NEU_NUM1 * NEU_NUM2 * DATA_BATCH));

    delete [] processed_labels;
	
    double loss;
    double * logloss = new double [1000]();
    double * loss_values = new double [DATA_BATCH];
	double * correct_rate = new double [VALID_BATCH];
    for(int i=0; i<VALID_BATCH; i++){
    	correct_rate[i] = 1;
    }

 	double cur_min = 1;
	int count=1;
	int batch_size = 0;
	int batch_num = (train_size - 1)/DATA_BATCH + 1;

	start = clock();

    DataLayer dataLayer(train_size * (NEIGHBOR + 1) * z, train_size * NEU_NUM2);
    dataLayer.input.data_d = gpu_processed_train;
    dataLayer.labels.data_d = gpu_processed_labels;

    Layer conv( (NEIGHBOR + 1) * z, (NEIGHBOR + 1) * P_NUM * FILTER_NUM, FILTER_NUM, re_size * FILTER_NUM, DATA_BATCH, false);

    Layer pooling(re_size * FILTER_NUM, re_size * FILTER_NUM, mre_size * FILTER_NUM, mre_size * FILTER_NUM, DATA_BATCH, false);
    
    Layer fulconnect(mre_size * FILTER_NUM, mre_size * FILTER_NUM * NEU_NUM1, NEU_NUM1, NEU_NUM1, DATA_BATCH, false);

    Layer out(NEU_NUM1, NEU_NUM1 * NEU_NUM2, NEU_NUM2, NEU_NUM2, DATA_BATCH, true);
    out.deltaW.data_d = gpu_out_deltaW;

    cudaDeviceSynchronize();
    int max_iter = 300;
    fprintf(stdout, "[Cube CNN training with MBGD Algorithm  Proportion of Training samples: %d%%  max_iter = %d] lr = %lf\n", 80, max_iter, LEARN_RATE);
	//creat CUDA streams
	cudaStream_t stream[DATA_BATCH];
	for(int i=0; i<DATA_BATCH; i++){
		cudaStreamCreate(&stream[i]);
	}    
	for (int j = 0; j < max_iter; j++ ){
		loss = 0;
        clock_t epoch_start = clock();
		for(int i0=0; i0<batch_num; i0++)
		{
			//compute the number of streams(or batch size)
			batch_size = DATA_BATCH;
			if((i0+1 == batch_num) && (train_size%DATA_BATCH != 0))
				batch_size = train_size%DATA_BATCH;
			
			for(int i1=0; i1<batch_size; i1++)
			{
				// forward propagation
                convolution<<<FILTER_NUM, re_size, (NEIGHBOR+1) * z * sizeof(double), stream[i1]>>>(i0*DATA_BATCH+i1,
                                                                                                    i1, 
                                                                                                    z,
                                                                                                    stride,
                                                                                                    re_size,
                                                                                                    dataLayer.input.data_d, 
                                                                                                    conv.weights.data_d, 
                                                                                                    conv.bias.data_d, 
                                                                                                    conv.output.data_d);

				maxpooling<<< FILTER_NUM, mre_size, 0, stream[i1] >>>( i1,
                                                                       re_size,
                                                                       mre_size, 
                                                                       conv.output.data_d, 
                                                                       pooling.output.data_d, 
                                                                       pooling.deltaB.data_d );
				
				fully_connect<<< NEU_NUM1, mre_size * FILTER_NUM, mre_size * FILTER_NUM * sizeof(double), stream[i1] >>>( i1, 
                                                                                                                          mre_size,
                                                                                                                          pooling.output.data_d, 
                                                                                                                          fulconnect.weights.data_d, 
                                                                                                                          fulconnect.bias.data_d, 
                                                                                                                          fulconnect.output.data_d );
				
				output<<< 1, NEU_NUM2, 0, stream[i1] >>>( i1, 
                                                          fulconnect.output.data_d, 
                                                          out.weights.data_d, 
                                                          out.bias.data_d, 
                                                          out.output.data_d );
										
				// backward propagation
				bp_output<<<1, NEU_NUM2, 0, stream[i1]>>>( iter, 
                                                           i0*DATA_BATCH+i1, 
                                                           i1, 
                                                           LEARN_RATE, 
                                                           dataLayer.labels.data_d, 
                                                           out.output.data_d, 
                                                           out.deltaB.data_d );
				
				bp_fully_connect<<<NEU_NUM1, NEU_NUM2, 0, stream[i1]>>>( i1, 
                                                                         LEARN_RATE, 
                                                                         out.weights.data_d, 
                                                                         out.deltaB.data_d, 
                                                                         out.deltaW.data_d, 
                                                                         fulconnect.output.data_d, 
                                                                         fulconnect.deltaB.data_d );
				
				bp_maxpooling<<< mre_size * FILTER_NUM, NEU_NUM1, 0, stream[i1] >>>( i1, 
                                                                                     LEARN_RATE, 
                                                                                     re_size, 
                                                                                     mre_size, 
                                                                                     pooling.output.data_d, 
                                                                                     pooling.deltaB.data_d,
                                                                                     pooling.deltaW.data_d,
                                                                                     fulconnect.weights.data_d,
                                                                                     fulconnect.deltaB.data_d, 
                                                                                     fulconnect.deltaW.data_d );
				
				bp_convolution<<< FILTER_NUM, (NEIGHBOR+1)*P_NUM, (NEIGHBOR+1) * z * sizeof(double), stream[i1] >>>( i0*DATA_BATCH+i1,
                                                                                                                     i1,
                                                                                                                     LEARN_RATE,
                                                                                                                     z,
                                                                                                                     stride,
                                                                                                                     mre_size,
                                                                                                                     re_size,
                                                                                                                     pooling.deltaB.data_d,
                                                                                                                     pooling.deltaW.data_d,
                                                                                                                     conv.deltaW.data_d,
                                                                                                                     conv.deltaB.data_d,
                                                                                                                     dataLayer.input.data_d );

			}

            cudaDeviceSynchronize();

            loss_function<<< 1, batch_size >>>( i0, 
                                                batch_size, 
                                                out.output.data_d, 
                                                dataLayer.labels.data_d, 
                                                gpu_loss_values );

            checkCudaErrors(cudaMemcpy(loss_values, gpu_loss_values, sizeof(double) * batch_size, cudaMemcpyDeviceToHost));
			
            cudaDeviceSynchronize();
			for( int j0 = 0; j0 < batch_size; j0 ++ )
            {
				loss = loss + loss_values[j0];
			}

			//update parameters
			update_output<<< NEU_NUM2, batch_size >>>( batch_size, 
                                                       LEARN_RATE, 
                                                       out.deltaB.data_d, 
                                                       out.bias.data_d );
			
			update_fully_connect<<<NEU_NUM1, NEU_NUM2>>>(batch_size, 
                                                         LEARN_RATE, 
                                                         out.weights.data_d, 
                                                         fulconnect.bias.data_d, 
                                                         out.deltaW.data_d, 
                                                         fulconnect.deltaB.data_d);
			
			update_maxpooling<<< mre_size * FILTER_NUM, NEU_NUM1 >>>( mre_size, 
                                                                      batch_size, 
                                                                      LEARN_RATE, 
                                                                      fulconnect.weights.data_d, 
                                                                      fulconnect.deltaW.data_d );
			
			update_convolution<<< FILTER_NUM, (NEIGHBOR + 1) * P_NUM >>>( batch_size, 
                                                                          re_size, 
                                                                          LEARN_RATE, 
                                                                          pooling.deltaW.data_d, 
                                                                          conv.deltaW.data_d, 
                                                                          conv.deltaB.data_d, 
                                                                          conv.weights.data_d, 
                                                                          conv.bias.data_d );
		
        }

        clock_t epoch_stop = clock();
        float epoch_time = float(epoch_stop - epoch_start) / CLOCKS_PER_SEC;
		double single_rate = loss/train_size;
       		logloss[j] = single_rate;
		
		fprintf(stdout,"[Cube CNN training with MBGD Algorithm  Proportion of Training Samples: %d%%  max_iter = %d  Execution time: %.3f sec] Epoch %d, loss = %lf;\n", 
                80, max_iter, epoch_time, j + 1, single_rate);
        	
		insert_line(correct_rate, single_rate);//insert current loss into the line
		double new_min = *min_element(correct_rate, correct_rate + VALID_BATCH);
        	if(cur_min > new_min){
            		cur_min = new_min;
		     	count = 1;
        	}
        	else{
            		count++;
        	}
        	if(count >= VALID_BATCH ) {
            		LEARN_RATE = LEARN_RATE * 0.9;
            		fprintf(stdout,"[Cube CNN training with MBGD Algorithm  Proportion of Training Samples: %d%%  max_iter = %d] lr = %lf\n",
                            80, max_iter, LEARN_RATE);
            		count = 1;
            		cur_min = new_min;
        	}
        	if(single_rate < MIN_ERR)
            		break;
	}

	fprintf(stdout,"[Cube CNN training with MBGD Algorithm  Proportion of Training Samples: %d%%  max_iter = %d ]", 80, max_iter);
	end = clock();
	tt = double(end - start);
	fprintf(stdout," Completed! Global Exesution time is %f sec\n", tt/CLOCKS_PER_SEC);

	start = clock();
	cudaDeviceSynchronize();
	checkCudaErrors(cudaMemcpy(conv.weights.data_h, conv.weights.data_d, sizeof(double) * (NEIGHBOR+1) * P_NUM * FILTER_NUM, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(conv.bias.data_h, conv.bias.data_d, sizeof(double) * FILTER_NUM, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(fulconnect.bias.data_h, fulconnect.bias.data_d, sizeof(double) * NEU_NUM1, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(out.bias.data_h, out.bias.data_d, sizeof(double) * NEU_NUM2, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(fulconnect.weights.data_h, fulconnect.weights.data_d, sizeof(double) * ome_num1, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(out.weights.data_h, out.weights.data_d, sizeof(double) * ome_num2, cudaMemcpyDeviceToHost));
	
	// Write the parameters into a mat file
	MATFile * pmatFile;
	pmatFile = matOpen("model/model.mat","w");
	mxArray * m1 = mxCreateDoubleMatrix((NEIGHBOR+1)*P_NUM, FILTER_NUM,mxREAL);
	memcpy((void *)mxGetPr(m1), (void *)conv.weights.data_h, sizeof(double) * (NEIGHBOR+1) * P_NUM * FILTER_NUM);
	matPutVariable(pmatFile, "filters", m1);

	mxArray * m2 = mxCreateDoubleMatrix(FILTER_NUM,1,mxREAL);
	memcpy((void *)mxGetPr(m2), (void *)conv.bias.data_h, sizeof(double) * FILTER_NUM);
	matPutVariable(pmatFile, "bias0", m2);

	mxArray * m3 = mxCreateDoubleMatrix(NEU_NUM1,mre_size * FILTER_NUM,mxREAL);
	memcpy((void *)mxGetPr(m3), (void *)fulconnect.weights.data_h, sizeof(double) * ome_num1);
	matPutVariable(pmatFile, "omega1", m3);

	mxArray * m4 = mxCreateDoubleMatrix(NEU_NUM1,1,mxREAL);
    memcpy((void *)mxGetPr(m4), (void *)fulconnect.bias.data_h, sizeof(double) * NEU_NUM1);
	matPutVariable(pmatFile, "bias1", m4);

	mxArray * m5 = mxCreateDoubleMatrix(NEU_NUM2,NEU_NUM1,mxREAL);
	memcpy((void *)mxGetPr(m5), (void *)out.weights.data_h, sizeof(double) * ome_num2);
	matPutVariable(pmatFile, "omega2", m5);

	mxArray * m6 = mxCreateDoubleMatrix(NEU_NUM2,1,mxREAL);
	memcpy((void *)mxGetPr(m6), (void *)out.bias.data_h, sizeof(double) * NEU_NUM2);
	matPutVariable(pmatFile, "bias2", m6);

    mxArray * m7 = mxCreateDoubleMatrix(300,1,mxREAL);
    memcpy((void *)mxGetPr(m7), (void *)logloss, sizeof(double) * 300);
    matPutVariable(pmatFile, "loss", m7);

	matClose(pmatFile);

    delete [] logloss;
    delete [] loss_values;
    delete [] correct_rate;

	for(int i=0; i<DATA_BATCH; i++){
		cudaStreamDestroy(stream[i]);
	}
	
	//test
	double right = 0;
	double count0 = 0;
    /*cudaStream_t testStream[test_size]; 
    for (size_t i=0; i<test_size; i++){
        cudaStreamCreate(&testStream[i]);
    }*/

	for (int i1=0; i1<test_size; i1++){
		convolution<<< FILTER_NUM, re_size, (NEIGHBOR + 1) * z * sizeof(double)/*, testStream[i1]*/ >>>( i1,
                                                                                                     0,
                                                                                                     z,
                                                                                                     stride,
                                                                                                     re_size,
                                                                                                     gpu_processed_test,
                                                                                                     conv.weights.data_d,
                                                                                                     conv.bias.data_d,
                                                                                                     conv.output.data_d );
		//cudaDeviceSynchronize();

		maxpooling<<< FILTER_NUM, mre_size, 0/*, testStream[i1]*/ >>>( 0,
                                                                   re_size,
                                                                   mre_size, 
                                                                   conv.output.data_d, 
                                                                   pooling.output.data_d, 
                                                                   pooling.deltaB.data_d );
		//cudaDeviceSynchronize();

		fully_connect<<< NEU_NUM1, mre_size * FILTER_NUM, mre_size * FILTER_NUM * sizeof(double)/*, testStream[i1]*/ >>>( 0, 
                                                                                                                      mre_size,
                                                                                                                      pooling.output.data_d, 
                                                                                                                      fulconnect.weights.data_d,
                                                                                                                      fulconnect.bias.data_d,
                                                                                                                      fulconnect.output.data_d );

		output<<< 1, NEU_NUM2, 0/*, testStream[i1]*/ >>>( 0,
                                                      fulconnect.output.data_d,
                                                      out.weights.data_d,
                                                      out.bias.data_d,
                                                      out.output.data_d );
		//cudaDeviceSynchronize();

		checkCudaErrors(cudaMemcpy(out.output.data_h, out.output.data_d, sizeof(double) * NEU_NUM2, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();

		right = count_err(test_labels, out.output.data_h, i1);
		count0 = count0 + right;
	}

    delete [] test_labels;

    cudaFree(gpu_processed_train);
    cudaFree(gpu_processed_test);
    cudaFree(gpu_processed_labels);
    cudaFree(gpu_out_deltaW);

    /*for (size_t i=0; i<test_size; i++){
        cudaStreamDestroy(testStream[i]);
    }*/
	end = clock();
	tt = float(end - start);
	fprintf(stdout, "[Cube CNN testing] Execution time is %.3fs. ", tt/CLOCKS_PER_SEC);
  
    return count0/test_size;
}


int main(int argc, char * argv[])
{
    fprintf(stdout, "[Cube CNN training with MBGD Algorithm] ");
  	if(!InitCUDA()){
		return 0;
	}
	printf("CUDA initialized.\n");

    fprintf(stdout, "[Cube CNN training with MBGD Algorithm] Available Device List: ");
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    int device;
    for (device = 0; device < deviceCount; ++ device)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d -- %s  ", device, deviceProp.name);
    }

    cout<<endl;
    cudaSetDevice(1);

	double *trainset,*trainlabels;
	if(argc!=2){
		fprintf(stderr, "2 input arguments required!");
		return 0;
	}

	MATFile * datamat = matOpen(argv[1], "r");
	mxArray * train = matGetVariable(datamat,"DataSet");
	mxArray * labels = matGetVariable(datamat,"labels");

	trainset = (double*)mxGetData(train);
	trainlabels = (double*)mxGetData(labels);

	const mwSize * dim;
	dim = mxGetDimensions(train);
	matClose(datamat);

	double correct = training(trainset, trainlabels, dim[0], dim[1], dim[2]);
	fprintf(stdout,"Correct Rate: %f%% \n", correct * 100);
    
    cudaDeviceReset();
	return 0;
}
