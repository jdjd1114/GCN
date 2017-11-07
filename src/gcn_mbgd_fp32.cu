#include <mat.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <matrix.h>
#include <iostream>
#include "cuda_util.h"
#include <cuda_runtime.h>
using namespace std;

const int KER_NUM = 20;//number of filters
const int P_NUM = 19;//number of layers of each convolution operation
const int LEAP = 2;//leap size
const int GP_NUM = 2;//size of each group
const int NEU_NUM1 = 100;//number of neurons in full connection layer
const int NEU_NUM2 = 13;//output layer
const int NEIGHBOR = 8;//number of neighbor pixels
float LEARN_RATE = 0.02;
const float MIN_ERR = 0.0001;
const int VALID_BATCH = 5;
const int MAX_MRE = 2000;
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
			if(prop.major>=1){                                                                                                                                      break;
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
__device__ void copy_data_to_share(float * data, float data_tmp[1][MAX_MRE],int tid, int offset,int head,int length){
	for(int i=tid*offset; i<(tid+1)*offset && (i < length); i++){
		data_tmp[0][i] = data[i+head];
	}
	__syncthreads();

}

/*__device__ void copy_data_to_shared(float * data, float * data_tmp, int head, int length){
	for(int i=0;i<length;i++){
		data_tmp[i] = data[i + head];
	}
}*/

//forward convolution
__global__ static void convolution(int iter,int data_id,int batch_id, float * train, float * kernel, float * re, float * bias,int z,int re_size)
{

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	if(tid<re_size && bid < KER_NUM)
	{
		__shared__ float train_tmp[1][MAX_MRE];
		int head = data_id*(NEIGHBOR+1)*z;
		int length = (NEIGHBOR+1)*z;
		int offset = (length - 1)/re_size + 1;
		copy_data_to_share(train,train_tmp,tid,offset,head,(NEIGHBOR+1)*z);
		__shared__ float kernel_tmp[1][MAX_MRE];
		head = bid * (NEIGHBOR+1)*P_NUM;
		length = (NEIGHBOR+1)*P_NUM;
		offset = (length - 1)/re_size + 1;
		copy_data_to_share(kernel,kernel_tmp,tid,offset,head,length);
		__syncthreads();

		float mid = 0;
		for(int i=0;i<(NEIGHBOR+1)*P_NUM;i++){
			mid = mid + kernel_tmp[0][i] * train_tmp[0][tid*(NEIGHBOR+1)*LEAP + i];
		}
		mid = mid + bias[bid];
		re[tid + bid*re_size + batch_id*re_size*KER_NUM] = 2/(1 + (1/exp(2*mid))) - 1;
	}
}

//forward maxpooling
__global__ static void maxpooling(int iter, int batch_id, float * re, float * mre,int * mre_index,int re_size,int mre_num){
	/*int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	}*/
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid<mre_num && bid<KER_NUM){
		float mid;
		int mid_index, head, tail;
		head = tid*GP_NUM + bid*re_size + batch_id*re_size*KER_NUM;
		tail = (tid+1)*GP_NUM + bid*re_size + batch_id*re_size*KER_NUM;
		mid = re[head];
		mid_index = head;
		for(int i=head; i<tail && (i<(bid+1)*re_size+batch_id*re_size*KER_NUM); i++){
			if(mid < re[i]){
				mid = re[i];
				mid_index=i;
			}
		}
		mre[tid + bid*mre_num + batch_id*mre_num*KER_NUM] = mid;
		mre_index[tid + bid*mre_num + batch_id*mre_num*KER_NUM] = mid_index;
	}
}

//forward full connection
__global__ static void fully_connect(int iter,int batch_id, float * mre, float * omega, float * bias, float * F1,int mre_size){

	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid<mre_size && bid<NEU_NUM1){
		__shared__ float ner[1][MAX_MRE];
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

//forward output
__global__ static void output(int iter, int batch_id, float * F1, float * omega2, float * bias, float * O2){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;
	int id = tid + iter * threadNum;

	if(id < NEU_NUM2){
		//copy F1 to shared memory
		__shared__ float F1_tmp[1][MAX_MRE];
		int offset = (NEU_NUM1-1)/NEU_NUM2 + 1;
		copy_data_to_share(F1, F1_tmp, id, offset, batch_id*NEU_NUM1, NEU_NUM1);
		__syncthreads();
		__shared__ float O2_tmp[1][NEU_NUM2];

		//compute the output of a neuron
		float mid = 0;
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


//backward output
__global__ static void bp_output(int iter, int train_idx, int batch_id, float * labels, float * O2, float * delta_L_z)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;
	int id = tid + iter * threadNum;
	if(id < NEU_NUM2){
		delta_L_z[id + batch_id*NEU_NUM2] = (O2[id + batch_id*NEU_NUM2] - labels[id + train_idx * NEU_NUM2])/NEU_NUM2;
	}
}
//backward fully connect
__global__ static void bp_fully_connect(int iter, int batch_id, float * omega2, float * F1, float * delta_L_z, float *delta_f_w, float * delta_f_z)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid<NEU_NUM2 && bid<NEU_NUM1){
		__shared__ float dfa[1][NEU_NUM2];
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

		//delta_f_a[bid] = dfa[0][0];
		delta_f_z[bid + batch_id*NEU_NUM1] = dfa[0][0] * (1 + F1[bid +batch_id*NEU_NUM1]) * (1 - F1[bid + batch_id*NEU_NUM1]);
	}
}
//backward maxpooling
__global__ static void bp_maxpooling(int iter, int mre_size, int re_size, int batch_id, int * mre_index, float * omega1, float * mre, float * delta_f_z, float * delta_m_w,  float * delta_22)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid<NEU_NUM1 && bid<mre_size){
		__shared__ float mid[1][NEU_NUM1];
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
			delta_22[mre_index[bid + batch_id*mre_size] + batch_id*re_size*KER_NUM] = mid[0][0] * (1 + mre[bid + batch_id*mre_size]) * (1 - mre[bid + batch_id*mre_size]);
	}
}

//backward convolution, update kernel
__global__ static void bp_convolution(int iter, int i0, int batch_id, int z, int mre_num,int re_size, int * mre_index, float * delta_22, float * delta_k_w, float * delta_k_b, float * data)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid < ((NEIGHBOR+1)*P_NUM) && (bid < KER_NUM))
	{
		int head = i0*(NEIGHBOR+1)*z;
		int length = (NEIGHBOR+1)*z;
		int offset = (length - 1)/((NEIGHBOR+1)*P_NUM) + 1;
		__shared__ float train_tmp[1][MAX_MRE];
		copy_data_to_share(data, train_tmp, tid, offset, head, length);
		__syncthreads();
		//extern __shared__ float train_tmp[];
                //copy_data_to_shared(data, train_tmp, (NEIGHBOR+1)*z*i0, (NEIGHBOR+1)*z);
		__shared__ float delta_kw[1][(NEIGHBOR+1)*P_NUM];
		delta_kw[0][tid] = 0;

		int idx, n, i, h;
		float mid = 0;
		for(i=0; i<mre_num; i++){
			idx = mre_index[i + bid*mre_num + batch_id*mre_num*KER_NUM];
			n = idx % re_size;
			h = n*(NEIGHBOR+1)*LEAP;
			delta_kw[0][tid] = delta_kw[0][tid] + delta_22[idx + batch_id*re_size*KER_NUM] * train_tmp[0][tid + h];
			mid = mid + delta_22[idx + batch_id*re_size*KER_NUM];
		}

		delta_k_w[tid + bid*(NEIGHBOR+1)*P_NUM + batch_id*(NEIGHBOR+1)*P_NUM*KER_NUM] =delta_kw[0][tid]/mre_num;
		
		if(tid < 1)
			delta_k_b[bid + batch_id*KER_NUM] = (mid/mre_num);
		
		
	}
}

//update function, output layer
__global__ static void update_output(int iter,int batch_size, float LEARN_RATE, float * delta_L_z, float * bias2)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid < batch_size && bid<NEU_NUM2){
		__shared__ float mid[1][DATA_BATCH];
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
//update function, fully connect layer
__global__ static void update_fully_connect(int batch_size, float LEARN_RATE, float * omega2,float * bias1,float *delta_f_w, float * delta_f_z)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid<NEU_NUM2 && bid<NEU_NUM1){
		float mid0 = 0, mid1 = 0;
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
//update function, maxpooling layer
__global__ static void update_maxpooling(int iter, int mre_size, int batch_size, float LEARN_RATE, float * omega1, float * delta_m_w)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid<NEU_NUM1 && bid<mre_size){
		float mid=0;
		for(int i=0; i<batch_size; i++){
			mid = mid + delta_m_w[tid + bid*NEU_NUM1 + i*mre_size*NEU_NUM1];
		}
		
		omega1[tid + bid*NEU_NUM1] = omega1[tid + bid*NEU_NUM1] - LEARN_RATE*mid/batch_size;
	}
}

//update function, convolution layer
__global__ static void update_convolution(int batch_size, int re_size, float LEARN_RATE, float * delta_22, float * delta_k_w, float * delta_k_b, float * kernel,float * bias0)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid < ((NEIGHBOR+1)*P_NUM) && (bid < KER_NUM))
	{
		float mid0 = 0, mid1 = 0;
		for(int i=0; i<batch_size; i++){
			mid0 = mid0 + delta_k_w[tid + bid*(NEIGHBOR+1)*P_NUM +i*(NEIGHBOR+1)*P_NUM*KER_NUM];
			mid1 = mid1 + delta_k_b[bid + i*KER_NUM];
		}
		kernel[tid + bid*(NEIGHBOR+1)*P_NUM] = kernel[tid + bid*(NEIGHBOR+1)*P_NUM] - LEARN_RATE*mid0/batch_size;
		
		if(tid < 1){
			bias0[bid] = bias0[bid] - LEARN_RATE*mid1/batch_size;
		}	
	}
}


__global__ static void loss_function(int batch_id, int batch_size, float * O2, float * labels, float * loss_values_sum)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float tmp = 0.0;
    if (tid < batch_size){
        __shared__ float loss_values[1][DATA_BATCH];
        for (int i=0; i < NEU_NUM2; i++){
            tmp = tmp + labels[i + (batch_id * DATA_BATCH + tid) * NEU_NUM2] * log(O2[i + tid * NEU_NUM2]) + 
                   (1 - labels[i + (batch_id * DATA_BATCH + tid) * NEU_NUM2]) * log(1 - O2[i + tid * NEU_NUM2]);
        }
    
        loss_values[0][tid] = -tmp/NEU_NUM2;
        __syncthreads();

        int length = batch_size;
        int offset = (length - 1)/2 + 1;
    
        while (length >= 2){
            if(tid + offset < length){
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
__global__ static void preprocessing(int iter, float * data, int * train_index, float * processed_data, int x, int y, int z, int train_size)
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

float lossfunction(float * output, float * labels, int idx){
	float l = 0;
	int i;
	for(i=0; i<NEU_NUM2; i++){
		l = l + labels[i + idx*NEU_NUM2] * log(output[i]) + (1 - labels[i + idx*NEU_NUM2])*log(1 - output[i]); 
        //(output[i] - labels[i + idx*NEU_NUM2]) * (output[i] - labels[i + idx*NEU_NUM2]);
	}
	l = -l/NEU_NUM2;
	return l;
}


//compute correct rate
float count_err(float * test_labels, float * output, int test_idx)
{
	float right=0;
	float max =0;
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
void insert_line(float * a, float b){
	for(int i=1; i<VALID_BATCH; i++){
		a[i-1] = a[i];
	}
	a[VALID_BATCH-1] = b;
}

float max(float * a){
	float m=a[0];
	for(int i=1; i<VALID_BATCH; i++){
		if(m<a[i])
			m=a[i];
	}
	return m;
}
float min(float * a){
    float mini = a[0];
    for(int i=1; i<VALID_BATCH; i++){
        if(mini > a[i]){
            mini = a[i];
        }
    }
    return mini;
}
//shuffle
void shuffle(int * data, float * labels, int dim_row, int width){
	int index,  i;
	int temp;
	float tmp;
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
float training(float * data, double * labels, int x, int y, int z){
	clock_t start, end;
	start = clock();	
	float * gpu_data;//original hyperspectral image, saved in global memory
	float * gpu_processed_train;//extracted train samples
	float * gpu_processed_test;//extracted test samples
	//float * gpu_processed_valid;
	int * gpu_train_index;//index of train samples and their neighbors
	int * gpu_test_index;//index of test samples
	//int * gpu_valid_index;
	float * gpu_processed_labels;//encoded train labels

	//preprocessing
	int data_size = 0;
	int * data_index = new int [x*y];
	for(int i=0; i<x*y; i++){
		if(labels[i] != 0 ){
			data_index[data_size]=i;
			data_size ++;
		}
	}
    cout<< "data_size = " << data_size <<endl; //DH

	int test_size = (data_size-1)/5 + 1;
	//int valid_size = test_size;
	int train_size = data_size - test_size;
	fprintf(stdout,"train_size:%d  test_size:%d\n",train_size,test_size);
	int * train_index = new int [train_size * (NEIGHBOR + 1)];
	//int * valid_index = new int [valid_size * (NEIGHBOR + 1)];
	int * test_index = new int [test_size * (NEIGHBOR+1)];

	float * processed_labels = new float [train_size * NEU_NUM2]();
	float * test_labels = new float [test_size]();
	//float * valid_labels = new float [valid_size]();
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
			test_labels[te] = float(labels[data_index[i]]);
			te = te + 1;
		}
	}

	shuffle(train_index, processed_labels, (NEIGHBOR+1), train_size);//shuffle the samples in training set
	//fprintf(stdout,"train_size:%d\n",train_size);
	//fprintf(stdout,"train_index:%d %d %d %d\ntest_index:%d %d %d %d\nvalid_index:%d %d %d %d\n",train_index[0],train_index[1],train_index[2],train_index[3],test_index[0],test_index[1],test_index[2],test_index[3],valid_index[0],valid_index[1],valid_index[2],valid_index[3]);
	//fprintf(stdout,"train labels:\n");
	//for(int i=0; i<NEU_NUM2; i++){
	//	fprintf(stdout,"%lf ",processed_labels[i]);
	//}
	//fprintf(stdout,"\n");
	//fprintf(stdout,"test label:%lf",test_labels[0]);
	//fprintf(stdout,"valid label:%lf",valid_labels[0]);
	//int * train_index = new int [train_size * (NEIGHBOR + 1)];

	//malloc GPU memory, copy data to GPU
	SAFE_CALL(cudaMalloc((void **) &gpu_data, sizeof(float) * x * y * z));
	SAFE_CALL(cudaMemcpy(gpu_data, data, sizeof(float)* x * y * z, cudaMemcpyHostToDevice));

	SAFE_CALL(cudaMalloc((void **) &gpu_train_index, sizeof(int) * train_size * (NEIGHBOR+1)));
	SAFE_CALL(cudaMemcpy(gpu_train_index, train_index, sizeof(int) * train_size * (NEIGHBOR+1), cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMalloc((void **) &gpu_test_index, sizeof(int) * test_size * (NEIGHBOR+1)));
	SAFE_CALL(cudaMemcpy(gpu_test_index, test_index, sizeof(int) * test_size * (NEIGHBOR+1), cudaMemcpyHostToDevice));
	//SAFE_CALL(cudaMalloc((void **) &gpu_valid_index, sizeof(int) * valid_size * (NEIGHBOR+1)));
	//SAFE_CALL(cudaMemcpy(gpu_valid_index, valid_index, sizeof(int) * valid_size * (NEIGHBOR+1), cudaMemcpyHostToDevice));

	//SAFE_CALL(cudaMalloc((void **) &gpu_processed_valid, sizeof(float) * valid_size * (NEIGHBOR+1) * z));
	SAFE_CALL(cudaMalloc((void **) &gpu_processed_test, sizeof(float) * test_size * (NEIGHBOR+1) * z));
	SAFE_CALL(cudaMalloc((void **) &gpu_processed_train, sizeof(float) * train_size * (NEIGHBOR+1) * z));

	int gridsize = 64;
	int blocksize = 1024;

    int iter = 0;
    preprocessing<<<gridsize, blocksize>>>(iter, gpu_data, gpu_train_index, gpu_processed_train, x, y, z, train_size);
    preprocessing<<<gridsize, blocksize>>>(iter, gpu_data, gpu_test_index, gpu_processed_test, x, y, z, test_size);
    end = clock();
	float tt = float(end - start);
	fprintf(stdout,"Preprocessing Done. (%lfs)\n",tt/CLOCKS_PER_SEC);

	//SAFE_CALL(cudaMemcpy(processed_train, gpu_processed_train, sizeof(float) * train_size * (NEIGHBOR+1) * z, cudaMemcpyDeviceToHost));
	//SAFE_CALL(cudaMemcpy(processed_test, gpu_processed_test, sizeof(float) * test_size * (NEIGHBOR+1) * z, cudaMemcpyDeviceToHost));
    //SAFE_CALL(cudaMemcpy(processed_valid, gpu_processed_valid, sizeof(float) * valid_size * (NEIGHBOR+1) * z, cudaMemcpyDeviceToHost));

	SAFE_CALL(cudaFree(gpu_data));
	SAFE_CALL(cudaFree(gpu_train_index));
	SAFE_CALL(cudaFree(gpu_test_index));
	//SAFE_CALL(cudaFree(gpu_valid_index));
	//cudaDeviceSynchronize();

	//fprintf(stdout,"Processed train data:%lf %lf %lf %lf\n",processed_train[0],processed_train[1],processed_train[2],processed_train[3]);
	//fprintf(stdout,"Processed test data:%lf %lf %lf %lf\n",processed_test[0],processed_test[1],processed_test[2],processed_test[3]);
   //fprintf(stdout,"processed valid data:%lf %lf %lf %lf\n",processed_valid[0],processed_valid[1],processed_valid[2],processed_valid[3]);

    
	//forward pass
	float * kernel = new float [(NEIGHBOR+1)*P_NUM*KER_NUM];

	//random initialize 
	for(int i=0; i<(NEIGHBOR+1)*P_NUM*KER_NUM; i++){
		kernel[i] = 2*(rand()/(float)(RAND_MAX)) - 1 ;
		kernel[i] = kernel[i]/20;
		if(kernel[i] == 0 )
			kernel[i] = 0.005;
	}
	//fprintf(stdout,"kernel:%lf %lf %lf %lf\n",kernel[0], kernel[1], kernel[2], kernel[3]);
	
	//count number of convolutional results
	int re_size = 0;
	for (int i=0; i+P_NUM-1<z; i+=LEAP){
		re_size ++;
	}

	//float * re = new float [re_size * KER_NUM];
	fprintf(stdout,"re_size:%d\n",re_size);

	int mre_num = (re_size-1)/GP_NUM + 1;
	fprintf(stdout,"mre_num:%d\n",mre_num);
	int mre_size = mre_num * KER_NUM;
	int ome_num1 = mre_num * KER_NUM * NEU_NUM1;//number of weights in full connection layer
	int ome_num2 = NEU_NUM1 * NEU_NUM2;//number of weights in output layer
	
	float * gpu_kernel;
	float * gpu_bias0;
	float * gpu_re;//results of concolution layer
	float * gpu_mre;//results of maxpooling layer
	int * gpu_mre_index;//index of max value in each group
	float * gpu_omega1;//weighs of full connection layer
	float * gpu_F1;//outputs of full connection layer
	float * gpu_bias1;
	float * gpu_omega2;
	float * gpu_O2;
	float * gpu_bias2;
	float * gpu_delta_Lz;
	float * gpu_delta_fz;
	float * gpu_delta_fw;
	float * gpu_delta_mw;	
	float * gpu_delta_22;
	float * gpu_delta_kb;
	float * gpu_delta_kw;
    float * gpu_loss_values;

	//copy labels to GPU
	SAFE_CALL(cudaMalloc((void**) &gpu_processed_labels, sizeof(float) * train_size * NEU_NUM2));
	SAFE_CALL(cudaMemcpy(gpu_processed_labels,processed_labels,sizeof(float) * train_size * NEU_NUM2,cudaMemcpyHostToDevice));
	//copy filters to GPU
	SAFE_CALL(cudaMalloc((void**) &gpu_kernel,sizeof(float) * (NEIGHBOR+1) * P_NUM * KER_NUM));
	SAFE_CALL(cudaMemcpy(gpu_kernel,kernel,sizeof(float) * (NEIGHBOR+1) * P_NUM * KER_NUM,cudaMemcpyHostToDevice));
	//malloc GPU memory for convolutional results
	SAFE_CALL(cudaMalloc((void **) &gpu_re,sizeof(float) * re_size * KER_NUM * DATA_BATCH));
	//malloc GPU memory for delta_Lz
	SAFE_CALL(cudaMalloc((void **) &gpu_delta_Lz, sizeof(float) * NEU_NUM2 * DATA_BATCH));

	//delta_f in full connection layer
	SAFE_CALL(cudaMalloc((void **) &gpu_delta_fz, sizeof(float) * NEU_NUM1 * DATA_BATCH));
	SAFE_CALL(cudaMalloc((void **) &gpu_delta_fw, sizeof(float) * NEU_NUM1 * NEU_NUM2 * DATA_BATCH));

	//maxpooling
	SAFE_CALL(cudaMalloc((void **) &gpu_delta_mw, sizeof(float) * mre_size * NEU_NUM1 * DATA_BATCH));

	//delta in input layer
	SAFE_CALL(cudaMalloc((void **) &gpu_delta_22,sizeof(float) * re_size * KER_NUM * DATA_BATCH));
	SAFE_CALL(cudaMalloc((void **) &gpu_delta_kw, sizeof(float) * (NEIGHBOR+1) * P_NUM * KER_NUM * DATA_BATCH));
	SAFE_CALL(cudaMalloc((void **) &gpu_delta_kb, sizeof(float) * KER_NUM * DATA_BATCH));
    SAFE_CALL(cudaMalloc((void **) &gpu_loss_values, sizeof(float) * 2 ));

	float * omega1 = new float [ome_num1];
	float * omega2 = new float [ome_num2];
	float * bias0 = new float [KER_NUM];
	float * bias1 = new float [NEU_NUM1];
	float * bias2 = new float [NEU_NUM2];

	//Initialize omega1
	for(int i=0; i<ome_num1; i++){
		omega1[i] = 2 * (rand()/(float)(RAND_MAX)) - 1;
		omega1[i] = omega1[i]/20;
	        if(omega1[i] == 0)
			omega1[i] = 0.01;
	}
	//initialize bias0
	for(int i=0; i<KER_NUM; i++){
		bias0[i] = 2*(rand()/(float)(RAND_MAX)) - 1;
		bias0[i] = bias0[i]/20;
	}
	//initialize bias1
	for(int i=0; i<NEU_NUM1; i++){
		bias1[i] = 2*(rand()/(float)(RAND_MAX)) - 1;
		bias1[i] = bias1[i]/20;
	}

	//initialize omega2
	for(int i=0; i<ome_num2; i++){
		omega2[i] = 2 * (rand()/(float)(RAND_MAX)) - 1;
		omega2[i] = omega2[i]/20;
		if(omega2[i] ==0)
			omega2[i] = 0.01;
	}
	//fprintf(stdout, "Bias1: %lf %lf %lf\n",bias1[0],bias1[1],bias1[2]);
	//initialize bias2
	for(int i=0; i<NEU_NUM2; i++){
		bias2[i] = 2*(rand()/(float)(RAND_MAX)) - 1;
		bias2[i] = bias2[i]/20;
	}
	//fprintf(stdout, "Bias2: %lf %lf %lf\n",bias2[0],bias2[1],bias2[2]);

    //malloc GPU memory for network parameters and intermediate results, copy the initialized values to GPU
	SAFE_CALL(cudaMalloc((void **) &gpu_mre, sizeof(float) * mre_num * KER_NUM * DATA_BATCH));
	SAFE_CALL(cudaMalloc((void **) &gpu_mre_index, sizeof(int) * mre_num * KER_NUM * DATA_BATCH));
	SAFE_CALL(cudaMalloc((void **) &gpu_omega1, sizeof(float) * ome_num1));
	SAFE_CALL(cudaMalloc((void **) &gpu_omega2, sizeof(float) * ome_num2));
	SAFE_CALL(cudaMalloc((void **) &gpu_F1, sizeof(float) * NEU_NUM1 * DATA_BATCH));
	SAFE_CALL(cudaMalloc((void **) &gpu_O2, sizeof(float) * NEU_NUM2 * DATA_BATCH));
	SAFE_CALL(cudaMalloc((void **) &gpu_bias0, sizeof(float) * KER_NUM));
	SAFE_CALL(cudaMalloc((void **) &gpu_bias1, sizeof(float) * NEU_NUM1));
	SAFE_CALL(cudaMalloc((void **) &gpu_bias2, sizeof(float) * NEU_NUM2));
	SAFE_CALL(cudaMemcpy(gpu_omega1, omega1, sizeof(float) * ome_num1, cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMemcpy(gpu_omega2, omega2, sizeof(float) * ome_num2, cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMemcpy(gpu_bias0, bias0, sizeof(float) * KER_NUM, cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMemcpy(gpu_bias1, bias1, sizeof(float) * NEU_NUM1, cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMemcpy(gpu_bias2, bias2, sizeof(float) * NEU_NUM2, cudaMemcpyHostToDevice));

	//float * delta_22 = new float [re_size * KER_NUM];
	//float * bias0 = new float [KER_NUM];
	float * O2 = new float [NEU_NUM2 * DATA_BATCH];//save output value of the network on Host
	//float * lz = new float [NEU_NUM2];
	float * loss_values = new float [DATA_BATCH];
    float loss;
    float * logloss = new float [1000]();
	float * correct_rate = new float [VALID_BATCH];
    	for(int i=0; i<VALID_BATCH; i++){
        	correct_rate[i] = 1;
    	}

	//clock_t * t = new clock_t [8]();
 	float cur_min = 1;
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
	
	for(int j=0; j<300; j++){
		loss = 0;
		for(int i0=0; i0<batch_num; i0++)
		{
			//compute the number of streams(or batch size)
			batch_size = DATA_BATCH;
			if((i0+1 == batch_num) && (train_size%DATA_BATCH != 0))
				batch_size = train_size%DATA_BATCH;
			
			for(int i1=0; i1<batch_size; i1++)
			{
				//forward convolution
				convolution<<<KER_NUM,re_size,0,stream[i1]>>>(iter,i0*DATA_BATCH+i1,i1,gpu_processed_train,gpu_kernel,gpu_re,gpu_bias0,z,re_size);
				//cudaDeviceSynchronize();	
				
				//forward maxpooling
				maxpooling<<<KER_NUM,mre_num,0,stream[i1]>>>(iter,i1,gpu_re,gpu_mre,gpu_mre_index,re_size,mre_num);
				//cudaDeviceSynchronize();
				//forward full connection
				fully_connect<<<NEU_NUM1,mre_size,0,stream[i1]>>>(iter,i1,gpu_mre,gpu_omega1,gpu_bias1,gpu_F1,mre_size);
				//cudaDeviceSynchronize();
				//forward output
				output<<<1,NEU_NUM2,0,stream[i1]>>>(iter,i1,gpu_F1,gpu_omega2,gpu_bias2,gpu_O2);
				//cudaDeviceSynchronize();
				//SAFE_CALL(cudaMemcpyAsync(O2+i1*NEU_NUM2, gpu_O2+i1*NEU_NUM2, sizeof(float) * NEU_NUM2, cudaMemcpyDeviceToHost, stream[i1]));
				//cudaDeviceSynchronize();
				//float single_loss = lossfunction(O2, processed_labels, i0*DATA_BATCH+i1);
				//loss = loss + single_loss;
				
				//backward output
				bp_output<<<1,NEU_NUM2,0,stream[i1]>>>(iter,i0*DATA_BATCH+i1,i1,gpu_processed_labels,gpu_O2,gpu_delta_Lz);
				//cudaDeviceSynchronize();
				//backward full connection
				bp_fully_connect<<<NEU_NUM1,NEU_NUM2,0,stream[i1]>>>(iter,i1,gpu_omega2,gpu_F1,gpu_delta_Lz,gpu_delta_fw,gpu_delta_fz);
				//cudaDeviceSynchronize();
				//backward maxpooling
				bp_maxpooling<<<mre_size,NEU_NUM1,0,stream[i1]>>>(iter,mre_size,re_size,i1,gpu_mre_index,gpu_omega1,gpu_mre,gpu_delta_fz,gpu_delta_mw,gpu_delta_22);
				//cudaDeviceSynchronize();
				//backward convolution
				bp_convolution<<<KER_NUM,(NEIGHBOR+1)*P_NUM,0,stream[i1]>>>(iter,i0*DATA_BATCH+i1,i1,z,mre_num,re_size,gpu_mre_index,gpu_delta_22,gpu_delta_kw,gpu_delta_kb,gpu_processed_train);
				//cudaDeviceSynchronize();
			}
			//cudaDeviceSynchronize();
			/*for(int j0=0; j0<batch_size; j0++){
				loss = loss + lossfunction(O2+j0*NEU_NUM2, processed_labels, i0*DATA_BATCH+j0);
			}*/
            float loss_tmp = 0;
            loss_function<<<1, DATA_BATCH>>>(i0, batch_size, gpu_O2, gpu_processed_labels, gpu_loss_values);
            SAFE_CALL(cudaMemcpy(&loss_tmp, gpu_loss_values, sizeof(float), cudaMemcpyDeviceToHost));
            loss += loss_tmp;
            //for (int j0=0; j0<batch_size; j0++)
            //    loss = loss_values[0];

			//update parameters of output layer
			update_output<<<NEU_NUM2,batch_size>>>(iter, batch_size, LEARN_RATE, gpu_delta_Lz, gpu_bias2);
			//cudaDeviceSynchronize();
			//update parameters of full connection layer
			update_fully_connect<<<NEU_NUM1,NEU_NUM2>>>(batch_size, LEARN_RATE, gpu_omega2, gpu_bias1, gpu_delta_fw, gpu_delta_fz);
			//cudaDeviceSynchronize();
			//update parameters of maxpooling layer
			update_maxpooling<<<mre_size,NEU_NUM1>>>(iter, mre_size, batch_size, LEARN_RATE, gpu_omega1, gpu_delta_mw);
			//cudaDeviceSynchronize();
			//update parameters of convolutional layer
			update_convolution<<<KER_NUM,(NEIGHBOR+1)*P_NUM>>>(batch_size, re_size, LEARN_RATE, gpu_delta_22, gpu_delta_kw, gpu_delta_kb, gpu_kernel, gpu_bias0);
			//cudaDeviceSynchronize();
		}
		
		float single_rate = loss/train_size;
       		logloss[j] = single_rate;
		//single_rate = single_rate/valid_size;
		fprintf(stdout,"Iteration %d,	loss = %lf;\n",j+1,single_rate);
        	
		insert_line(correct_rate,single_rate);//insert current loss into the line
		float new_min = min(correct_rate);
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
	tt = float(end - start);
	fprintf(stdout,"Exesution time of training:%lfs\n",tt/CLOCKS_PER_SEC);

	start = clock();
	//cudaDeviceSynchronize();
	SAFE_CALL(cudaMemcpy(kernel, gpu_kernel, sizeof(float) * (NEIGHBOR+1) * P_NUM * KER_NUM, cudaMemcpyDeviceToHost));
	SAFE_CALL(cudaMemcpy(bias0, gpu_bias0, sizeof(float) * KER_NUM, cudaMemcpyDeviceToHost));
	SAFE_CALL(cudaMemcpy(bias1, gpu_bias1, sizeof(float) * NEU_NUM1, cudaMemcpyDeviceToHost));
	SAFE_CALL(cudaMemcpy(bias2, gpu_bias2, sizeof(float) * NEU_NUM2, cudaMemcpyDeviceToHost));
	SAFE_CALL(cudaMemcpy(omega1, gpu_omega1, sizeof(float) * ome_num1, cudaMemcpyDeviceToHost));
	SAFE_CALL(cudaMemcpy(omega2, gpu_omega2, sizeof(float) * ome_num2, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	//fprintf(stdout,"kernel:%lf %lf %lf %lf\n",kernel[0], kernel[1], kernel[2], kernel[3]);

	//write the network parameters into a mat file
	MATFile * pmatFile;
	pmatFile = matOpen("model.mat","w");
	mxArray * m1 = mxCreateDoubleMatrix((NEIGHBOR+1)*P_NUM,KER_NUM,mxREAL);
	memcpy((void *)mxGetPr(m1), (void *)kernel, sizeof(float) * (NEIGHBOR+1) * P_NUM * KER_NUM);
	matPutVariable(pmatFile, "kernel", m1);

	mxArray * m2 = mxCreateDoubleMatrix(KER_NUM,1,mxREAL);
	memcpy((void *)mxGetPr(m2), (void *)bias0, sizeof(float) * KER_NUM);
	matPutVariable(pmatFile, "bias0", m2);

	mxArray * m3 = mxCreateDoubleMatrix(NEU_NUM1,mre_size,mxREAL);
	memcpy((void *)mxGetPr(m3), (void *)omega1, sizeof(float) * ome_num1);
	matPutVariable(pmatFile, "omega1", m3);

	mxArray * m4 = mxCreateDoubleMatrix(NEU_NUM1,1,mxREAL);
       	memcpy((void *)mxGetPr(m4), (void *)bias1, sizeof(float) * NEU_NUM1);
	matPutVariable(pmatFile, "bias1", m4);

	mxArray * m5 = mxCreateDoubleMatrix(NEU_NUM2,NEU_NUM1,mxREAL);
	memcpy((void *)mxGetPr(m5), (void *)omega2, sizeof(float) * ome_num2);
	matPutVariable(pmatFile, "omega2", m5);

	mxArray * m6 = mxCreateDoubleMatrix(NEU_NUM2,1,mxREAL);
	memcpy((void *)mxGetPr(m6), (void *)bias2, sizeof(float) * NEU_NUM2);
	matPutVariable(pmatFile, "bias2", m6);

    	mxArray * m7 = mxCreateDoubleMatrix(300,1,mxREAL);
    	memcpy((void *)mxGetPr(m7), (void *)logloss, sizeof(float) * 300);
    	matPutVariable(pmatFile, "loss", m7);

	matClose(pmatFile);
	//fprintf(stdout,"mre:%lf %lf %lf\n",mre[0],mre[1],mre[2]);
	//fprintf(stdout,"mre_index:%d %d %d\n",mre_index[0],mre_index[1],mre_index[2]);

	//fprintf(stdout,"F1 Output:%lf %lf; %lf %lf\n",F1[0],F1[1],F1[98],F1[99]);
	//fprintf(stdout,"O2 Output:%lf %lf; %lf %lf\n",O2[0],O2[1],O2[18],O2[19]);
	//end = clock();
	//tt = float(end - start);
	//fprintf(stdout, "Using time of writeback:%lfs\n",tt/CLOCKS_PER_SEC);
	//destroy streams
	for(int i=0; i<DATA_BATCH; i++){
		cudaStreamDestroy(stream[i]);
	}
	
	//test
	float right = 0;
	float count0 = 0;
	for (int i1=0; i1<test_size; i1++){
		int iter = 0;
		convolution<<<KER_NUM,re_size>>>(iter,i1,0,gpu_processed_test,gpu_kernel,gpu_re,gpu_bias0,z,re_size);
		//cudaDeviceSynchronize();

		maxpooling<<<KER_NUM,mre_num>>>(iter,0,gpu_re,gpu_mre,gpu_mre_index,re_size,mre_num);
		//cudaDeviceSynchronize();

		fully_connect<<<NEU_NUM1,mre_size>>>(iter,0,gpu_mre,gpu_omega1,gpu_bias1,gpu_F1,mre_size);
		//cudaDeviceSynchronize();

		output<<<1,NEU_NUM2>>>(iter,0,gpu_F1,gpu_omega2,gpu_bias2,gpu_O2);
		//cudaDeviceSynchronize();

		SAFE_CALL(cudaMemcpy(O2, gpu_O2, sizeof(float) * NEU_NUM2, cudaMemcpyDeviceToHost));
		//cudaDeviceSynchronize();

		//fprintf(stdout,"\n");
		right = count_err(test_labels, O2, i1);
		count0 = count0 + right;
	}
	end = clock();
	tt = float(end - start);
	fprintf(stdout,"Execution time of testing:%lfs\n",tt/CLOCKS_PER_SEC);
    cout<< "count0 = " <<count0<<endl;
	return count0/test_size;
}


int main(int argc, char * argv[])
{
  	if(!InitCUDA()){
		return 0;
	}
	printf("CUDA initialized.\n");

	clock_t start,end;

	float * trainset;
    double * trainlabels;
	if(argc!=2){
		fprintf(stderr, "2 input arguments required!");
		return 0;
	}

	MATFile * datamat = matOpen(argv[1], "r");
	mxArray * train = matGetVariable(datamat,"DataSet");
	mxArray * labels = matGetVariable(datamat,"labels");

	trainset = (float*)mxGetData(train);
	trainlabels = (double*)mxGetData(labels);

	const mwSize * dim;
	dim = mxGetDimensions(train);
	matClose(datamat);

	start = clock();
	float correct = training(trainset, trainlabels, dim[0], dim[1], dim[2]);
	end = clock();
	fprintf(stdout,"Correct Rate:%lf(300 iterations, train:test=4:1)\n",correct);
	float usetime = float(end - start);
	fprintf(stdout, "Execution time of the whole program:%lfs\n",usetime/CLOCKS_PER_SEC);
	return 0;
}
