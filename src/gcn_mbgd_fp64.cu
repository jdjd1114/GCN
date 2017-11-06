#include <mat.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <matrix.h>
#include <iostream>
#include "cuda_util.h"
#include <cuda_runtime.h>
#include <algorithm>
using namespace std;

const int FILTER_NUM = 20;//number of filters
const int P_NUM = 19;//number of layers of each convolution operation
const int LEAP = 2;//leap size
const int GP_NUM = 2;//size of each group
const int NEU_NUM1 = 100;//number of neurons in full connection layer
const int NEU_NUM2 = 13;//output layer
const int NEIGHBOR = 8;//number of neighbor pixels
double LEARN_RATE = 0.2;
const double MIN_ERR = 0;
const int VALID_BATCH = 5;
const int MAX_MRE = 2000;
const int DATA_BATCH = 10;//batch size

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
	cudaSetDevice(i);
	return true;
}


//copy data to shared memory
__device__ void copy_data_to_shared(double * data, double data_tmp[][NEIGHBOR+1], int tid, int row, int head){
	for(size_t i = 0; i < row; i++){
		data_tmp[i][tid] = data[i * (NEIGHBOR+1) + tid + head];
	}
	__syncthreads();

}

__device__ void copy_data_to_share(double * data, double data_tmp[1][MAX_MRE], int tid, int offset, int head, int length)
{
    for(size_t i = tid * offset; i < (tid + 1) * offset && (i < length); i++)
    {
        data_tmp[0][i] = data[i + head];
    }
    __syncthreads();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// forward propagation kernels
// forward convolution
__global__ static void convolution(int iter,int data_id,int batch_id, double * train,double * filters, double * re,double * bias, int z,int re_size)
{

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	if(tid<re_size && bid < FILTER_NUM)
	{
		__shared__ double train_tmp[1][MAX_MRE];
        int head = data_id * (NEIGHBOR + 1) * z;
		int length = (NEIGHBOR+1) * z;
        int offset = (length - 1)/re_size + 1;
		copy_data_to_share(train,train_tmp, tid, offset, head, (NEIGHBOR+1) * z);
        
		__shared__ double filters_tmp[1][MAX_MRE];
		head = bid * (NEIGHBOR+1)*P_NUM;
		length = (NEIGHBOR+1)*P_NUM;
		offset = (length - 1)/re_size + 1;
		copy_data_to_share(filters, filters_tmp,tid,offset,head,length);
		__syncthreads();

		double mid = 0;
		for(size_t i = 0; i < (NEIGHBOR + 1) * P_NUM; i++){
            mid = mid + filters_tmp[0][i] * train_tmp[0][tid * (NEIGHBOR+1) * LEAP + i];
		}
		mid = mid + bias[bid];
		re[tid + bid * re_size + batch_id * re_size * FILTER_NUM] = 2/(1 + (1/exp(2*mid))) - 1;
	}
}

// forward maxpooling
__global__ static void maxpooling(int iter,int batch_id,double * re,double * mre,int * mre_index,int re_size,int mre_num)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid < mre_num && bid < FILTER_NUM){
		double mid;
		int mid_index, head, tail;
		head = tid * GP_NUM + bid * re_size + batch_id * re_size * FILTER_NUM;
		tail = (tid+1) * GP_NUM + bid * re_size + batch_id * re_size * FILTER_NUM;
		mid = re[head];
		mid_index = head;
		for(int i=head; i<tail && (i<(bid+1)*re_size+batch_id*re_size*FILTER_NUM); i++){
			if(mid < re[i]){
				mid = re[i];
				mid_index=i;
			}
		}
		mre[tid + bid*mre_num + batch_id*mre_num*FILTER_NUM] = mid;
		mre_index[tid + bid*mre_num + batch_id*mre_num*FILTER_NUM] = mid_index;
	}
}

// forward fully connection
__global__ static void fully_connect(int iter,int batch_id,double * mre,double * omega,double * bias,double * F1,int mre_size){

	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid<mre_size && bid<NEU_NUM1){
		__shared__ double ner[1][MAX_MRE];
		ner[0][tid] = omega[bid + tid*NEU_NUM1] * mre[tid + batch_id*mre_size];
		__syncthreads();//waiting for other threads

		int length = mre_size;
		int offset = (length - 1)/2 + 1;

		while(length >= 2)
		{
			if(tid + offset < length)
			{
				ner[0][tid] = ner[0][tid] + ner[0][tid + offset];
			}
			offset = (offset - 1)/2 + 1;
			length = (length - 1)/2 + 1;
			__syncthreads();
		}

		F1[bid + batch_id*NEU_NUM1] = 2/(1 + 1/exp((ner[0][0] + bias[bid]) * 2)) - 1;
	}
}

// forward output
__global__ static void output(int iter, int batch_id, double * F1, double * omega2, double * bias, double * O2){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;
	int id = tid + iter * threadNum;

	if(id < NEU_NUM2){
		//copy F1 to shared memory
		__shared__ double F1_tmp[1][MAX_MRE];
		int offset = (NEU_NUM1-1)/NEU_NUM2 + 1;
		copy_data_to_share(F1, F1_tmp, id, offset, batch_id*NEU_NUM1, NEU_NUM1);
		__syncthreads();
		__shared__ double O2_tmp[1][NEU_NUM2];

		//compute the output of a neuron
		double mid = 0;
		for(int i=0; i<NEU_NUM1; i++){
			mid = mid + omega2[id + i*NEU_NUM2] * F1_tmp[0][i];
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
__global__ static void bp_output(int iter, int train_idx, int batch_id, double LEARN_RATE, double * labels, double * O2, double * delta_L_z)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;
	int id = tid + iter * threadNum;
	if(id < NEU_NUM2){
		delta_L_z[id + batch_id*NEU_NUM2] = (O2[id + batch_id*NEU_NUM2] - labels[id + train_idx * NEU_NUM2])/NEU_NUM2;
	}
}

// fully_connect
__global__ static void bp_fully_connect(int iter, int batch_id, double LEARN_RATE, double * omega2, double * F1, double * delta_L_z, double *delta_f_w, double * delta_f_z)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid<NEU_NUM2 && bid<NEU_NUM1){
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
__global__ static void bp_maxpooling(int iter, int mre_size, int re_size, int batch_id, double LEARN_RATE, int * mre_index, double * omega1, double * mre, double * delta_f_z, double * delta_m_w,  double * delta_22)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid<NEU_NUM1 && bid<mre_size){
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

		delta_m_w[tid + bid*NEU_NUM1 + batch_id*mre_size*NEU_NUM1] = mre[bid + batch_id*mre_size] * delta_f_z[tid + batch_id*NEU_NUM1];

		if(tid < 1)
			delta_22[mre_index[bid + 
                     batch_id*mre_size] +
                     batch_id*re_size*FILTER_NUM] = mid[0][0] * (1 + mre[bid + batch_id*mre_size]) * 
                                                    (1 - mre[bid + batch_id*mre_size]);
	}
}

// convolutional layer
__global__ static void bp_convolution(int iter, int i0, int batch_id, double LEARN_RATE, int z, int mre_num,int re_size, int * mre_index, double * delta_22, double * delta_k_w, double * delta_k_b, double * data)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid < ((NEIGHBOR+1)*P_NUM) && (bid < FILTER_NUM))
	{
		int head = i0*(NEIGHBOR+1)*z;
		int length = (NEIGHBOR+1)*z;
		int offset = (length - 1)/((NEIGHBOR+1)*P_NUM) + 1;
		__shared__ double train_tmp[1][MAX_MRE];
		copy_data_to_share(data, train_tmp, tid, offset, head, length);
		__syncthreads();
        __shared__ double delta_kw[1][(NEIGHBOR+1)*P_NUM];
		delta_kw[0][tid] = 0;

		int idx, n, i, h;
		double mid = 0;
		for(i=0; i<mre_num; i++){
			idx = mre_index[i + bid*mre_num + batch_id*mre_num*FILTER_NUM];
			n = idx % re_size;
			h = n*(NEIGHBOR+1)*LEAP;
			delta_kw[0][tid] = delta_kw[0][tid] + delta_22[idx + batch_id*re_size*FILTER_NUM] * train_tmp[0][tid + h];
			mid = mid + delta_22[idx + batch_id*re_size*FILTER_NUM];
		}

		delta_k_w[tid + bid*(NEIGHBOR+1)*P_NUM + batch_id*(NEIGHBOR+1)*P_NUM*FILTER_NUM] =delta_kw[0][tid]/mre_num;
		
		if(tid < 1)
			delta_k_b[bid + batch_id*FILTER_NUM] = (mid/mre_num);
		
		
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// update params kernels
// output layer
__global__ static void update_output(int iter,int batch_size, double LEARN_RATE, double * delta_L_z, double * bias2)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid < batch_size && bid<NEU_NUM2){
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
			bias2[bid] = bias2[bid] - mid[0][0]*LEARN_RATE/batch_size;
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
		omega2[tid + bid*NEU_NUM2] = omega2[tid + bid*NEU_NUM2] - LEARN_RATE * mid0/batch_size;
		
		if(tid < 1){
			bias1[bid] = bias1[bid] - LEARN_RATE * mid1 / batch_size;
		}
	}
}
// maxpooling layer
__global__ static void update_maxpooling(int iter, int mre_size, int batch_size, double LEARN_RATE, double * omega1, double * delta_m_w)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid<NEU_NUM1 && bid<mre_size){
		double mid=0;
		for(int i=0; i<batch_size; i++){
			mid = mid + delta_m_w[tid + bid*NEU_NUM1 + i*mre_size*NEU_NUM1];
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

__global__ static void loss_function(int batch_id, int batch_size, double * O2, double * labels, double * loss_values_sum)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    double tmp = 0.0;
    if(tid < batch_size){
        __shared__ double loss_values[1][DATA_BATCH];

        for(size_t i=0; i<NEU_NUM2; i++){
            tmp = tmp + labels[i + (batch_id * DATA_BATCH + tid) * NEU_NUM2] * log(O2[i + tid * NEU_NUM2]) +
                  (1 - labels[i + (batch_id * DATA_BATCH + tid) * NEU_NUM2]) * log(1 - O2[i + tid * NEU_NUM2]);
        }

        loss_values[0][tid] = -tmp/NEU_NUM2;
        __syncthreads();

        int length = batch_size;
        int offset = (length - 1)/2 + 1;

        while(length >= 2)
        {
            if(tid + offset < length)
            {
                loss_values[0][tid] = loss_values[0][tid] + loss_values[0][tid + offset];
            }

            length = (length - 1)/2 + 1;
            offset = (offset - 1)/2 + 1;
            __syncthreads();
        }

        loss_values_sum[0] = loss_values[0][0];
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
	fprintf(stdout,"train_size:%d  test_size:%d\n",train_size,test_size);
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
	SAFE_CALL(cudaMalloc((void **) &gpu_data, sizeof(double) * x * y * z));
	SAFE_CALL(cudaMemcpy(gpu_data, data, sizeof(double)* x * y * z, cudaMemcpyHostToDevice));

	SAFE_CALL(cudaMalloc((void **) &gpu_train_index, sizeof(int) * train_size * (NEIGHBOR+1)));
	SAFE_CALL(cudaMemcpy(gpu_train_index, train_index, sizeof(int) * train_size * (NEIGHBOR+1), cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMalloc((void **) &gpu_test_index, sizeof(int) * test_size * (NEIGHBOR+1)));
	SAFE_CALL(cudaMemcpy(gpu_test_index, test_index, sizeof(int) * test_size * (NEIGHBOR+1), cudaMemcpyHostToDevice));

	SAFE_CALL(cudaMalloc((void **) &gpu_processed_test, sizeof(double) * test_size * (NEIGHBOR+1) * z));
    SAFE_CALL(cudaMalloc((void **) &gpu_processed_train, sizeof(double) * train_size * (NEIGHBOR+1) *z));

    delete [] data_index;
    delete [] train_index;
    delete [] test_index;

    int gridsize = 64;
    int blocksize = 512;
	int iter=0;

	processing<<<gridsize,blocksize>>>(iter, gpu_data, gpu_train_index, gpu_processed_train, x, y, z, train_size);
	processing<<<gridsize,blocksize>>>(iter, gpu_data, gpu_test_index, gpu_processed_test, x, y, z, test_size);
	//processing<<<gridsize,blocksize>>>(iter, gpu_data, gpu_valid_index, gpu_processed_valid, x, y, z, valid_size);

	//cudaDeviceSynchronize();
	end = clock();
	double tt = double(end - start);
	fprintf(stdout,"Preprocessing Done. (%lfs)\n",tt/CLOCKS_PER_SEC);

	SAFE_CALL(cudaFree(gpu_data));
	SAFE_CALL(cudaFree(gpu_train_index));
	SAFE_CALL(cudaFree(gpu_test_index));
	//cudaDeviceSynchronize();
    
	//forward pass
	double * filters = new double [(NEIGHBOR+1)*P_NUM*FILTER_NUM];

	//random initialize 
	for(int i=0; i<(NEIGHBOR+1)*P_NUM*FILTER_NUM; i++){
		filters[i] = 2*(rand()/(double)(RAND_MAX)) - 1 ;
		filters[i] = filters[i]/20;
		if(filters[i] == 0 )
			filters[i] = 0.005;
	}
	
	//count number of convolutional results
	int re_size = 0;
	for (int i=0; i+P_NUM-1<z; i+=LEAP){
		re_size ++;
	}

	
	fprintf(stdout,"re_size:%d\n",re_size);

	int mre_num = (re_size-1)/GP_NUM + 1;
	fprintf(stdout,"mre_num:%d\n",mre_num);
	int mre_size = mre_num * FILTER_NUM;
	int ome_num1 = mre_num * FILTER_NUM * NEU_NUM1;//number of weights in full connection layer
	int ome_num2 = NEU_NUM1 * NEU_NUM2;//number of weights in output layer
	
	double * gpu_filters;
	double * gpu_bias0;
	double * gpu_re;//results of concolution layer
	double * gpu_mre;//results of maxpooling layer
	int * gpu_mre_index;//index of max value in each group
	double * gpu_omega1;//weighs of full connection layer
	double * gpu_F1;//outputs of full connection layer
	double * gpu_bias1;
	double * gpu_omega2;
	double * gpu_O2;
	double * gpu_bias2;
	double * gpu_delta_Lz;
	double * gpu_delta_fz;
	double * gpu_delta_fw;
	double * gpu_delta_mw;	
	double * gpu_delta_22;
	double * gpu_delta_kb;
	double * gpu_delta_kw;
    double * gpu_loss_values;

	//copy labels to GPU
	SAFE_CALL(cudaMalloc((void**) &gpu_processed_labels, sizeof(double) * train_size * NEU_NUM2));
	SAFE_CALL(cudaMemcpy(gpu_processed_labels,processed_labels,sizeof(double) * train_size * NEU_NUM2,cudaMemcpyHostToDevice));
	//copy filters to GPU
	SAFE_CALL(cudaMalloc((void**) &gpu_filters,sizeof(double) * (NEIGHBOR+1) * P_NUM * FILTER_NUM));
	SAFE_CALL(cudaMemcpy(gpu_filters, filters, sizeof(double) * (NEIGHBOR+1) * P_NUM * FILTER_NUM,cudaMemcpyHostToDevice));
	//malloc GPU memory for convolutional results
	SAFE_CALL(cudaMalloc((void **) &gpu_re,sizeof(double) * re_size * FILTER_NUM * DATA_BATCH));
	//malloc GPU memory for delta_Lz
	SAFE_CALL(cudaMalloc((void **) &gpu_delta_Lz, sizeof(double) * NEU_NUM2 * DATA_BATCH));

	//delta_f in full connection layer
	SAFE_CALL(cudaMalloc((void **) &gpu_delta_fz, sizeof(double) * NEU_NUM1 * DATA_BATCH));
	SAFE_CALL(cudaMalloc((void **) &gpu_delta_fw, sizeof(double) * NEU_NUM1 * NEU_NUM2 * DATA_BATCH));

	//maxpooling
	SAFE_CALL(cudaMalloc((void **) &gpu_delta_mw, sizeof(double) * mre_size * NEU_NUM1 * DATA_BATCH));

	//delta in input layer
	SAFE_CALL(cudaMalloc((void **) &gpu_delta_22,sizeof(double) * re_size * FILTER_NUM * DATA_BATCH));
	SAFE_CALL(cudaMalloc((void **) &gpu_delta_kw, sizeof(double) * (NEIGHBOR+1) * P_NUM * FILTER_NUM * DATA_BATCH));
	SAFE_CALL(cudaMalloc((void **) &gpu_delta_kb, sizeof(double) * FILTER_NUM * DATA_BATCH));
    SAFE_CALL(cudaMalloc((void **) &gpu_loss_values, sizeof(double) * 2));

    delete [] processed_labels;
	double * omega1 = new double [ome_num1];
	double * omega2 = new double [ome_num2];
	double * bias0 = new double [FILTER_NUM];
	double * bias1 = new double [NEU_NUM1];
	double * bias2 = new double [NEU_NUM2];

	//Initialize omega1
	for(int i=0; i<ome_num1; i++){
		omega1[i] = 2 * (rand()/(double)(RAND_MAX)) - 1;
		omega1[i] = omega1[i]/20;
	        if(omega1[i] == 0)
			omega1[i] = 0.01;
	}
	//initialize bias0
    for(int i=0; i<FILTER_NUM; i++){
		bias0[i] = 2*(rand()/(double)(RAND_MAX)) - 1;
		bias0[i] = bias0[i]/20;
	}
	//initialize bias1
	for(int i=0; i<NEU_NUM1; i++){
		bias1[i] = 2*(rand()/(double)(RAND_MAX)) - 1;
		bias1[i] = bias1[i]/20;
	}

	//initialize omega2
	for(int i=0; i<ome_num2; i++){
		omega2[i] = 2 * (rand()/(double)(RAND_MAX)) - 1;
		omega2[i] = omega2[i]/20;
		if(omega2[i] ==0)
			omega2[i] = 0.01;
	}

	//initialize bias2
	for(int i=0; i<NEU_NUM2; i++){
		bias2[i] = 2*(rand()/(double)(RAND_MAX)) - 1;
		bias2[i] = bias2[i]/20;
	}

    //malloc GPU memory for network parameters and intermediate results, copy the initialized values to GPU
	SAFE_CALL(cudaMalloc((void **) &gpu_mre, sizeof(double) * mre_num * FILTER_NUM * DATA_BATCH));
	SAFE_CALL(cudaMalloc((void **) &gpu_mre_index, sizeof(int) * mre_num * FILTER_NUM * DATA_BATCH));
	SAFE_CALL(cudaMalloc((void **) &gpu_omega1, sizeof(double) * ome_num1));
	SAFE_CALL(cudaMalloc((void **) &gpu_omega2, sizeof(double) * ome_num2));
	SAFE_CALL(cudaMalloc((void **) &gpu_F1, sizeof(double) * NEU_NUM1 * DATA_BATCH));
	SAFE_CALL(cudaMalloc((void **) &gpu_O2, sizeof(double) * NEU_NUM2 * DATA_BATCH));
	SAFE_CALL(cudaMalloc((void **) &gpu_bias0, sizeof(double) * FILTER_NUM));
	SAFE_CALL(cudaMalloc((void **) &gpu_bias1, sizeof(double) * NEU_NUM1));
	SAFE_CALL(cudaMalloc((void **) &gpu_bias2, sizeof(double) * NEU_NUM2));
	SAFE_CALL(cudaMemcpy(gpu_omega1, omega1, sizeof(double) * ome_num1, cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMemcpy(gpu_omega2, omega2, sizeof(double) * ome_num2, cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMemcpy(gpu_bias0, bias0, sizeof(double) * FILTER_NUM, cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMemcpy(gpu_bias1, bias1, sizeof(double) * NEU_NUM1, cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMemcpy(gpu_bias2, bias2, sizeof(double) * NEU_NUM2, cudaMemcpyHostToDevice));



	double * O2 = new double [NEU_NUM2 * DATA_BATCH];//save output value of the network on Host
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
	int batch_num = (train_size - 1)/DATA_BATCH + 1;//count how many bathes are needed to complete the whole training set
	fprintf(stdout,"batch_num:%d\n",batch_num);
	start = clock();
	//creat CUDA streams
	cudaStream_t stream[DATA_BATCH];
	for(int i=0; i<DATA_BATCH; i++){
		cudaStreamCreate(&stream[i]);
	}
	
	for(int j=0; j<30; j++){
		loss = 0;
		for(int i0=0; i0<batch_num; i0++)
		{
			//compute the number of streams(or batch size)
			batch_size = DATA_BATCH;
			if((i0+1 == batch_num) && (train_size%DATA_BATCH != 0))
				batch_size = train_size%DATA_BATCH;
			
			for(int i1=0; i1<batch_size; i1++)
			{
				// forward propagation
				convolution<<<FILTER_NUM,re_size,0,stream[i1]>>>(iter,i0*DATA_BATCH+i1,i1,gpu_processed_train,gpu_filters,gpu_re,gpu_bias0,z,re_size);
				
				maxpooling<<<FILTER_NUM,mre_num,0,stream[i1]>>>(iter,i1,gpu_re,gpu_mre,gpu_mre_index,re_size,mre_num);
				
				fully_connect<<<NEU_NUM1,mre_size,0,stream[i1]>>>(iter,i1,gpu_mre,gpu_omega1,gpu_bias1,gpu_F1,mre_size);
				
				output<<<1,NEU_NUM2,0,stream[i1]>>>(iter,i1,gpu_F1,gpu_omega2,gpu_bias2,gpu_O2);
										
				// backward propagation
				bp_output<<<1,NEU_NUM2,0,stream[i1]>>>(iter,i0*DATA_BATCH+i1,i1,LEARN_RATE,gpu_processed_labels,gpu_O2,gpu_delta_Lz);
				
				bp_fully_connect<<<NEU_NUM1,NEU_NUM2,0,stream[i1]>>>(iter,i1,LEARN_RATE,gpu_omega2,gpu_F1,gpu_delta_Lz,gpu_delta_fw,gpu_delta_fz);
				
				bp_maxpooling<<<mre_size,NEU_NUM1,0,stream[i1]>>>(iter,mre_size,re_size,i1,LEARN_RATE,gpu_mre_index,gpu_omega1,gpu_mre,gpu_delta_fz,gpu_delta_mw,gpu_delta_22);
				
				bp_convolution<<<FILTER_NUM,(NEIGHBOR+1)*P_NUM,0,stream[i1]>>>(iter,i0*DATA_BATCH+i1,i1,LEARN_RATE,z,mre_num,re_size,gpu_mre_index,gpu_delta_22,gpu_delta_kw,gpu_delta_kb,gpu_processed_train);
				
            }
            double loss_tmp = 0.0;
            loss_function<<<1, batch_size>>>(i0, batch_size, gpu_O2, gpu_processed_labels, gpu_loss_values);
            SAFE_CALL(cudaMemcpy(&loss_tmp, gpu_loss_values, sizeof(double), cudaMemcpyDeviceToHost));
			loss += loss_tmp;
			/*for(int j0=0; j0<batch_size; j0++){
				loss = loss + loss_values[j0];
			}*/

			//update parameters
			update_output<<<NEU_NUM2,batch_size>>>(iter, batch_size, LEARN_RATE, gpu_delta_Lz, gpu_bias2);
			
			update_fully_connect<<<NEU_NUM1,NEU_NUM2>>>(batch_size, LEARN_RATE, gpu_omega2, gpu_bias1, gpu_delta_fw, gpu_delta_fz);
			
			update_maxpooling<<<mre_size,NEU_NUM1>>>(iter, mre_size, batch_size, LEARN_RATE, gpu_omega1, gpu_delta_mw);
			
			update_convolution<<<FILTER_NUM,(NEIGHBOR+1)*P_NUM>>>(batch_size, re_size, LEARN_RATE, gpu_delta_22, gpu_delta_kw, gpu_delta_kb, gpu_filters, gpu_bias0);
			
        }
		
		double single_rate = loss/train_size;
       		logloss[j] = single_rate;
		
		fprintf(stdout,"Iteration %d,	loss = %lf;\n",j+1,single_rate);
        	
		insert_line(correct_rate,single_rate);//insert current loss into the line
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
            		fprintf(stdout,"LEARN_RATE:%lf\n",LEARN_RATE);
            		count = 1;
            		cur_min = new_min;
        	}
        	if(single_rate < MIN_ERR)
            		break;
	}

	fprintf(stdout,"Training completed!\n");
	end = clock();
	tt = double(end - start);
	fprintf(stdout,"Exesution time of training:%lfs\n",tt/CLOCKS_PER_SEC);

	start = clock();
	cudaDeviceSynchronize();
	SAFE_CALL(cudaMemcpy(filters, gpu_filters, sizeof(double) * (NEIGHBOR+1) * P_NUM * FILTER_NUM, cudaMemcpyDeviceToHost));
	SAFE_CALL(cudaMemcpy(bias0, gpu_bias0, sizeof(double) * FILTER_NUM, cudaMemcpyDeviceToHost));
	SAFE_CALL(cudaMemcpy(bias1, gpu_bias1, sizeof(double) * NEU_NUM1, cudaMemcpyDeviceToHost));
	SAFE_CALL(cudaMemcpy(bias2, gpu_bias2, sizeof(double) * NEU_NUM2, cudaMemcpyDeviceToHost));
	SAFE_CALL(cudaMemcpy(omega1, gpu_omega1, sizeof(double) * ome_num1, cudaMemcpyDeviceToHost));
	SAFE_CALL(cudaMemcpy(omega2, gpu_omega2, sizeof(double) * ome_num2, cudaMemcpyDeviceToHost));
	
	//write the network parameters into a mat file
	MATFile * pmatFile;
	pmatFile = matOpen("model.mat","w");
	mxArray * m1 = mxCreateDoubleMatrix((NEIGHBOR+1)*P_NUM, FILTER_NUM,mxREAL);
	memcpy((void *)mxGetPr(m1), (void *)filters, sizeof(double) * (NEIGHBOR+1) * P_NUM * FILTER_NUM);
	matPutVariable(pmatFile, "filters", m1);

	mxArray * m2 = mxCreateDoubleMatrix(FILTER_NUM,1,mxREAL);
	memcpy((void *)mxGetPr(m2), (void *)bias0, sizeof(double) * FILTER_NUM);
	matPutVariable(pmatFile, "bias0", m2);

	mxArray * m3 = mxCreateDoubleMatrix(NEU_NUM1,mre_size,mxREAL);
	memcpy((void *)mxGetPr(m3), (void *)omega1, sizeof(double) * ome_num1);
	matPutVariable(pmatFile, "omega1", m3);

	mxArray * m4 = mxCreateDoubleMatrix(NEU_NUM1,1,mxREAL);
       	memcpy((void *)mxGetPr(m4), (void *)bias1, sizeof(double) * NEU_NUM1);
	matPutVariable(pmatFile, "bias1", m4);

	mxArray * m5 = mxCreateDoubleMatrix(NEU_NUM2,NEU_NUM1,mxREAL);
	memcpy((void *)mxGetPr(m5), (void *)omega2, sizeof(double) * ome_num2);
	matPutVariable(pmatFile, "omega2", m5);

	mxArray * m6 = mxCreateDoubleMatrix(NEU_NUM2,1,mxREAL);
	memcpy((void *)mxGetPr(m6), (void *)bias2, sizeof(double) * NEU_NUM2);
	matPutVariable(pmatFile, "bias2", m6);

    mxArray * m7 = mxCreateDoubleMatrix(300,1,mxREAL);
    memcpy((void *)mxGetPr(m7), (void *)logloss, sizeof(double) * 300);
    matPutVariable(pmatFile, "loss", m7);

	matClose(pmatFile);

    delete [] filters;
    delete [] bias0;
    delete [] omega1;
    delete [] bias1;
    delete [] omega2;
    delete [] bias2;
    delete [] logloss;
    delete [] loss_values;
    delete [] correct_rate;

	for(int i=0; i<DATA_BATCH; i++){
		cudaStreamDestroy(stream[i]);
	}
	
	//test
	double right = 0;
	double count0 = 0;
    cudaStream_t testStream[test_size]; 
    for (size_t i=0; i<test_size; i++){
        cudaStreamCreate(&testStream[i]);
    }

	for (int i1=0; i1<test_size; i1++){
		int iter = 0;
		convolution<<<FILTER_NUM, re_size, 0, testStream[i1]>>>(iter,i1,0,gpu_processed_test,gpu_filters,gpu_re,gpu_bias0,z,re_size);
		//cudaDeviceSynchronize();

		maxpooling<<<FILTER_NUM, mre_num, 0, testStream[i1]>>>(iter,0,gpu_re,gpu_mre,gpu_mre_index,re_size,mre_num);
		//cudaDeviceSynchronize();

		fully_connect<<<NEU_NUM1,mre_size, 0, testStream[i1]>>>(iter,0,gpu_mre,gpu_omega1,gpu_bias1,gpu_F1,mre_size);
		//cudaDeviceSynchronize();

		output<<<1,NEU_NUM2, 0, testStream[i1]>>>(iter,0,gpu_F1,gpu_omega2,gpu_bias2,gpu_O2);
		//cudaDeviceSynchronize();

		SAFE_CALL(cudaMemcpy(O2, gpu_O2, sizeof(double) * NEU_NUM2, cudaMemcpyDeviceToHost));
		//cudaDeviceSynchronize();

		right = count_err(test_labels, O2, i1);
		count0 = count0 + right;
	}

    delete [] O2;
    delete [] test_labels;

    cudaFree(gpu_processed_test);
    cudaFree(gpu_mre);
    cudaFree(gpu_filters);
    cudaFree(gpu_re);
    cudaFree(gpu_bias0);
    cudaFree(gpu_mre_index);
    cudaFree(gpu_omega1);
    cudaFree(gpu_bias1);
    cudaFree(gpu_F1);
    cudaFree(gpu_omega2);
    cudaFree(gpu_bias2);
    cudaFree(gpu_O2);

    for (size_t i=0; i<test_size; i++){
        cudaStreamDestroy(testStream[i]);
    }
	end = clock();
	tt = double(end - start);
	fprintf(stdout,"Execution time of testing:%lfs\n",tt/CLOCKS_PER_SEC);
	return count0/test_size;
}


int main(int argc, char * argv[])
{
  	if(!InitCUDA()){
		return 0;
	}
	printf("CUDA initialized.\n");

	clock_t start,end;

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

	start = clock();
	double correct = training(trainset, trainlabels, dim[0], dim[1], dim[2]);
	end = clock();
	fprintf(stdout,"Correct Rate:%lf(300 iterations, train:test=4:1)\n",correct);
	double usetime = double(end - start);
	fprintf(stdout, "Execution time of the whole program:%lfs\n",usetime/CLOCKS_PER_SEC);
    cudaDeviceReset();
	return 0;
}
