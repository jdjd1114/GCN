#include <mat.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <matrix.h>
#include <iostream>
#include <algorithm>
#include "cuda_util.h"
#include <cuda_runtime.h>
using namespace std;

const int FILTERS_NUM = 20;//number of convolution filters
const int P_NUM = 19;//number of layers to be convoluted
const int LEAP = 2;//leap size
const int GP_NUM = 2;//number of each group of maxpooling layer
const int NEU_NUM1 = 100;
const int NEU_NUM2 = 13;//number of output layer neurons 
const int NEIGHBOR = 8;//number of neighbors
double LEARN_RATE = 0.007;
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
	cudaSetDevice(i);
	return true;
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
__global__ static void convolution(int i0,double * train,double * filters, double * re,double * bias,int z,int re_size)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	if(tid<re_size && bid < FILTERS_NUM)
	{
		extern __shared__ double train_tmp[];
		int head = i0*(NEIGHBOR+1)*z;
		int length = (NEIGHBOR+1)*z;
		int offset = (length - 1)/re_size + 1;
		copy_data_to_shared(train, train_tmp, tid, offset, head, (NEIGHBOR+1)*z);
		
        __shared__ double filters_tmp[(NEIGHBOR+1) * P_NUM];
		head = bid * (NEIGHBOR+1) * P_NUM;
		length = (NEIGHBOR+1)*P_NUM;
		offset = (length - 1)/re_size + 1;
		copy_data_to_shared(filters, filters_tmp,tid,offset,head,length);
		__syncthreads();

		double mid = 0;
		for(int i=0;i<(NEIGHBOR+1)*P_NUM;i++){
			mid = mid + filters_tmp[i] * train_tmp[tid*(NEIGHBOR+1)*LEAP + i];
		}
		mid = mid + bias[bid];
		re[tid + bid*re_size] = 2/(1 + (1/exp(2*mid))) - 1;
	}
}

__global__ static void maxpooling(double * re,double * mre,int * mre_index,int re_size,int mre_num){
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	if(tid<mre_num && bid<FILTERS_NUM){
		double mid;
		int mid_index, head, tail;
		head = tid*GP_NUM + bid*re_size;
		tail = (tid+1)*GP_NUM + bid*re_size ;
		mid = re[head];
		mid_index = head;
		for(int i=head; i<tail && (i<(bid+1)*re_size); i++){
			if(mid < re[i]){
				mid = re[i];
				mid_index=i;
			}
		}
		mre[tid + bid*mre_num] = mid;
		mre_index[tid + bid*mre_num] = mid_index;
	}
}

__global__ static void fully_connect(double * mre,double * omega,double * bias,double * F1,int mre_size)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid<mre_size && bid<NEU_NUM1){
		extern __shared__ double ner[];
		ner[tid] = omega[bid + tid*NEU_NUM1] * mre[tid];
		__syncthreads();

		int length = mre_size;
		int offset = (length - 1)/2 + 1;

		while(length >= 2)
		{
			if(tid + offset < length)
			{
				ner[tid] = ner[tid] + ner[tid + offset];
			}
			offset = (offset - 1)/2 + 1;
			length = (length - 1)/2 + 1;
			__syncthreads();
		}

		F1[bid] = 2/(1 + 1/exp((ner[0] + bias[bid]) * 2)) - 1;
	}
}

__global__ static void output(int tag, int train_idx, double * F1, double * omega2, double * bias, double * O2, double * labels, double * loss)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(id < NEU_NUM2)
    {
        __shared__ double F1_tmp[NEU_NUM1];
        size_t offset = (NEU_NUM1-1)/NEU_NUM2 + 1;
        copy_data_to_shared(F1, F1_tmp, id, offset, 0, NEU_NUM1);
		__syncthreads();
        
        __shared__ double O2_tmp[1][NEU_NUM2];
        double mid = 0;
        for(int i=0; i<NEU_NUM1; i++){
            mid = mid + omega2[id + i*NEU_NUM2] * F1_tmp[i];
        }
        O2[id] = exp(mid+ bias[id]);
        O2_tmp[0][id] = O2[id];
        __syncthreads();
        
        size_t length = NEU_NUM2;
        offset = (length - 1)/2 +1;
        while(length >= 2)
        {
            if(id + offset < length){
                O2_tmp[0][id] = O2_tmp[0][id] + O2_tmp[0][id + offset];
            }
            
            offset = (offset - 1)/2 + 1;
            length = (length - 1)/2 + 1;
             __syncthreads();
        }
		O2[id] = O2[id]/O2_tmp[0][0];

        if (tag == 1)
        {
            if (train_idx == 0)
                loss[0] = 0;

            __shared__ double loss_values[NEU_NUM2];
            loss_values[id] = labels[id + train_idx * NEU_NUM2] * log(O2[id]) + (1 - labels[id + train_idx * NEU_NUM2]) * log(1 - O2[id]);
            __syncthreads();
        
            length = NEU_NUM2;
            offset = (length - 1)/2 + 1;
            while (length >= 2){
                if(id + offset < length){
                    loss_values[id] = loss_values[id] + loss_values[id + offset];
                }

                offset = (offset - 1)/2 + 1;
                length = (length - 1)/2 + 1;
                __syncthreads();
            }

            loss[0] = loss[0] - loss_values[0]/NEU_NUM2;
        }
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// backward propagation
__global__ static void bp_output(int train_idx, double LEARN_RATE, double * labels, double * O2, double * bias2, double * delta_L_z)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
    if(id < NEU_NUM2)
    {
		delta_L_z[id] = (O2[id] - labels[id + train_idx * NEU_NUM2])/NEU_NUM2;
		bias2[id] = bias2[id] - delta_L_z[id]*LEARN_RATE;
	}
}

__global__ static void bp_fully_connect(double LEARN_RATE, double * omega2,double * bias1, double * F1, double * delta_L_z, double *delta_f_a, double * delta_f_z)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid<NEU_NUM2 && bid<NEU_NUM1){
		__shared__ double dfa[1][NEU_NUM2];
		dfa[0][tid] = omega2[tid + bid*NEU_NUM2] * delta_L_z[tid];
		__syncthreads();

		omega2[tid + bid*NEU_NUM2] = omega2[tid + bid*NEU_NUM2] - LEARN_RATE * F1[bid] * delta_L_z[tid];

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

		delta_f_a[bid] = dfa[0][0];
		delta_f_z[bid] = dfa[0][0] * (1 + F1[bid]) * (1 - F1[bid]);
		if(tid < 1){
			bias1[bid] = bias1[bid] - LEARN_RATE * delta_f_z[bid];
		}
	}
}

__global__ static void bp_maxpooling(int mre_size,double LEARN_RATE, int *mre_index, double * omega1,double *mre, double * delta_f_a, double * delta_f_z, double * delta_22)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid<NEU_NUM1 && bid<mre_size){
		__shared__ double mid[1][NEU_NUM1];
		mid[0][tid] = omega1[tid + bid*NEU_NUM1] * delta_f_z[tid];
		__syncthreads();

		omega1[tid + bid*NEU_NUM1] = omega1[tid + bid*NEU_NUM1] - LEARN_RATE*(mre[bid]*delta_f_z[tid]);

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

		if(tid < 1)
			delta_22[mre_index[bid]] = mid[0][0] * (1 + mre[bid]) * (1 - mre[bid]);
	}
}

__global__ static void bp_convolution(int i0, double LEARN_RATE, int z, int mre_num,int re_size, int * mre_index, double * delta_22, double * data, double * filters, double * bias0)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid < ((NEIGHBOR+1)*P_NUM) && (bid < FILTERS_NUM))
	{
		int head = i0*(NEIGHBOR+1)*z;
		int length = (NEIGHBOR+1)*z;
		int offset = (length - 1)/((NEIGHBOR+1)*P_NUM) + 1;
		extern __shared__ double train_tmp[];
		copy_data_to_shared(data, train_tmp, tid, offset, head, length);
		__syncthreads();

		__shared__ double delta_k_w[1][(NEIGHBOR+1)*P_NUM];
		delta_k_w[0][tid] = 0;

		int idx, n, i, h;
		double mid = 0;
		for(i=0; i<mre_num; i++){
			idx = mre_index[i + bid*mre_num];
			n = idx % re_size;
			h = n*(NEIGHBOR+1)*LEAP;
			delta_k_w[0][tid] = delta_k_w[0][tid] + delta_22[idx] * train_tmp[tid + h];
			mid = mid + delta_22[idx];
		}

		delta_k_w[0][tid] = delta_k_w[0][tid]/mre_num;
		filters[tid + bid*(NEIGHBOR+1)*P_NUM] = filters[tid + bid*(NEIGHBOR+1)*P_NUM] - LEARN_RATE*delta_k_w[0][tid];
		
		if(tid < 1)
			bias0[bid] = bias0[bid] - LEARN_RATE*(mid/mre_num);
		
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
	fprintf(stdout,"train_size:%d  test_size:%d\n", train_size, test_size);
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
    
    SAFE_CALL(cudaMalloc((void **) &gpu_data, sizeof(double) * x * y * z));
	SAFE_CALL(cudaMemcpy(gpu_data, data, sizeof(double)* x * y * z, cudaMemcpyHostToDevice));

	SAFE_CALL(cudaMalloc((void **) &gpu_train_index, sizeof(int) * train_size * (NEIGHBOR+1)));
	SAFE_CALL(cudaMemcpy(gpu_train_index, train_index, sizeof(int) * train_size * (NEIGHBOR+1), cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMalloc((void **) &gpu_test_index, sizeof(int) * test_size * (NEIGHBOR+1)));
	SAFE_CALL(cudaMemcpy(gpu_test_index, test_index, sizeof(int) * test_size * (NEIGHBOR+1), cudaMemcpyHostToDevice));


    SAFE_CALL(cudaMalloc((void **) &gpu_processed_test, sizeof(double) * test_size * (NEIGHBOR+1) * z));
    SAFE_CALL(cudaMalloc((void **) &gpu_processed_train, sizeof(double) * train_size * (NEIGHBOR+1) * z));
    
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
	fprintf(stdout,"Preprocessing Done. (%lfs)\n",tt/CLOCKS_PER_SEC);

	SAFE_CALL(cudaFree(gpu_data));
	SAFE_CALL(cudaFree(gpu_train_index));
	SAFE_CALL(cudaFree(gpu_test_index));
	cudaDeviceSynchronize();

	double * filters = new double [(NEIGHBOR+1) * P_NUM * FILTERS_NUM];
	for(int i=0; i<(NEIGHBOR+1)*P_NUM*FILTERS_NUM; i++){
		filters[i] = 2*(rand()/(double)(RAND_MAX)) - 1 ;
		filters[i] = filters[i]/55;
	}
	
    int re_size = 0;
	for (int i=0; i+P_NUM-1<z; i+=LEAP){
		re_size ++;
	}
    //fprintf(stdout,"re_size:%d\n",re_size);

	int mre_num = (re_size-1)/GP_NUM + 1;
	//fprintf(stdout,"mre_num:%d\n",mre_num);
	int mre_size = mre_num * FILTERS_NUM;
    int ome_num1 = mre_num * FILTERS_NUM * NEU_NUM1;
    int ome_num2 = NEU_NUM1 * NEU_NUM2;

    double * gpu_filters;
    double * gpu_bias0;
    double * gpu_re;
    double * gpu_mre;
    int * gpu_mre_index;
    double * gpu_omega1;
    double * gpu_F1;
    double * gpu_bias1;
    double * gpu_omega2;
    double * gpu_O2;
    double * gpu_bias2;
    double * gpu_delta_fa;
    double * gpu_delta_fz;
    double * gpu_delta_22;
    double * gpu_delta_Lz;
    double * gpu_loss;
    double * delta_22 = new double [re_size*FILTERS_NUM]();
    
    SAFE_CALL(cudaMalloc((void**) &gpu_processed_labels, sizeof(double) * train_size * NEU_NUM2));
	SAFE_CALL(cudaMemcpy(gpu_processed_labels,processed_labels,sizeof(double) * train_size * NEU_NUM2,cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMalloc((void**) &gpu_filters,sizeof(double) * (NEIGHBOR+1) * P_NUM * FILTERS_NUM));
    SAFE_CALL(cudaMemcpy(gpu_filters, filters, sizeof(double) * (NEIGHBOR+1) * P_NUM * FILTERS_NUM,cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMalloc((void **) &gpu_loss, sizeof(double) * 2));
    SAFE_CALL(cudaMalloc((void **) &gpu_re,sizeof(double) * re_size * FILTERS_NUM));
	SAFE_CALL(cudaMalloc((void **) &gpu_delta_Lz, sizeof(double) * NEU_NUM2));
	SAFE_CALL(cudaMalloc((void **) &gpu_delta_fa, sizeof(double) * NEU_NUM1));
	SAFE_CALL(cudaMalloc((void **) &gpu_delta_fz, sizeof(double) * NEU_NUM1));
	SAFE_CALL(cudaMalloc((void **) &gpu_delta_22,sizeof(double) * re_size * FILTERS_NUM));
	SAFE_CALL(cudaMemcpy(gpu_delta_22, delta_22, sizeof(double) * re_size * FILTERS_NUM, cudaMemcpyHostToDevice));

	double * omega1 = new double [ome_num1];
	double * omega2 = new double [ome_num2];
	double * bias0 = new double [FILTERS_NUM];
	double * bias1 = new double [NEU_NUM1];
	double * bias2 = new double [NEU_NUM2];

	for(int i=0; i<ome_num1; i++){
		omega1[i] = 2 * (rand()/(double)(RAND_MAX)) - 1;
		omega1[i] = omega1[i]/55;
	        if(omega1[i] == 0)
			omega1[i] = 0.01;
	}

	for(int i=0; i<FILTERS_NUM; i++){
		bias0[i] = 2*(rand()/(double)(RAND_MAX)) - 1;
		bias0[i] = bias0[i]/55;
	}

	for(int i=0; i<NEU_NUM1; i++){
		bias1[i] = 2*(rand()/(double)(RAND_MAX)) - 1;
		bias1[i] = bias1[i]/55;
	}

	for(int i=0; i<ome_num2; i++){
		omega2[i] = 2 * (rand()/(double)(RAND_MAX)) - 1;
		omega2[i] = omega2[i]/55;
		if(omega2[i] ==0)
			omega2[i] = 0.01;
	}
	
	for(int i=0; i<NEU_NUM2; i++){
		bias2[i] = 2*(rand()/(double)(RAND_MAX)) - 1;
		bias2[i] = bias2[i]/55;
	}

	SAFE_CALL(cudaMalloc((void **) &gpu_mre, sizeof(double) * mre_num * FILTERS_NUM));
    SAFE_CALL(cudaMalloc((void **) &gpu_mre_index, sizeof(int) * mre_num * FILTERS_NUM));
	SAFE_CALL(cudaMalloc((void **) &gpu_omega1, sizeof(double) * ome_num1));
    SAFE_CALL(cudaMalloc((void **) &gpu_omega2, sizeof(double) * ome_num2));
    SAFE_CALL(cudaMalloc((void **) &gpu_F1, sizeof(double) * NEU_NUM1));
    SAFE_CALL(cudaMalloc((void **) &gpu_O2, sizeof(double) * NEU_NUM2));
    SAFE_CALL(cudaMalloc((void **) &gpu_bias0, sizeof(double) * FILTERS_NUM));
    SAFE_CALL(cudaMalloc((void **) &gpu_bias1, sizeof(double) * NEU_NUM1));
    SAFE_CALL(cudaMalloc((void **) &gpu_bias2, sizeof(double) * NEU_NUM2));
    SAFE_CALL(cudaMemcpy(gpu_omega1, omega1, sizeof(double) * ome_num1, cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(gpu_omega2, omega2, sizeof(double) * ome_num2, cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(gpu_bias0, bias0, sizeof(double) * FILTERS_NUM, cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(gpu_bias1, bias1, sizeof(double) * NEU_NUM1, cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(gpu_bias2, bias2, sizeof(double) * NEU_NUM2, cudaMemcpyHostToDevice));

	double * O2 = new double [NEU_NUM2];
    double loss;
    double * logloss = new double [300]();
	double * correct_rate = new double [VALID_BATCH];
    for(int i=0; i<VALID_BATCH; i++){
        correct_rate[i] = 1;
    }

    double cur_min = 1;
	int count = 1;
    int tag = 1; // for training
	start = clock();
	for(int j=0; j<300; j++){
		loss = 0;
		for(int i0=0; i0<train_size; i0++)
        {
			convolution<<<FILTERS_NUM, re_size, (NEIGHBOR+1) * z * sizeof(double)>>>(i0,gpu_processed_train,gpu_filters, gpu_re,gpu_bias0,z,re_size);

		    maxpooling<<<FILTERS_NUM,mre_num>>>(gpu_re,gpu_mre,gpu_mre_index,re_size,mre_num);

			fully_connect<<<NEU_NUM1,mre_size, mre_size * sizeof(double)>>>(gpu_mre, gpu_omega1, gpu_bias1, gpu_F1, mre_size);
			
			output<<<1,NEU_NUM2>>>(tag, i0, gpu_F1, gpu_omega2, gpu_bias2, gpu_O2, gpu_processed_labels, gpu_loss);

			// backward propagation
			bp_output<<<1,NEU_NUM2>>>(i0, LEARN_RATE, gpu_processed_labels, gpu_O2, gpu_bias2, gpu_delta_Lz);
			
			bp_fully_connect<<<NEU_NUM1,NEU_NUM2>>>(LEARN_RATE, gpu_omega2, gpu_bias1, gpu_F1, gpu_delta_Lz, gpu_delta_fa, gpu_delta_fz);
			
			bp_maxpooling<<<mre_size,NEU_NUM1>>>(mre_size, LEARN_RATE, gpu_mre_index, gpu_omega1, gpu_mre, gpu_delta_fa, gpu_delta_fz, gpu_delta_22);
			
			bp_convolution<<<FILTERS_NUM, (NEIGHBOR+1)*P_NUM, (NEIGHBOR+1) * z * sizeof(double)>>>(i0, LEARN_RATE,z,mre_num,re_size,gpu_mre_index,gpu_delta_22,gpu_processed_train,gpu_filters, gpu_bias0);

		}

        SAFE_CALL(cudaMemcpy(&loss, gpu_loss, sizeof(double), cudaMemcpyDeviceToHost));
		double single_rate = loss/train_size;
        	logloss[j] = single_rate;
		if(single_rate < MIN_ERR)
			break;
		
		fprintf(stdout,"Iteration %d,	loss = %lf;\n",j+1,single_rate);
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
            		LEARN_RATE = LEARN_RATE * 0.9;
            		fprintf(stdout,"LEARN_RATE:%lf\n",LEARN_RATE);
            		count = 1;
            		cur_min = new_min;
        	}		
	}

	fprintf(stdout,"Training completed!\n");
	end = clock();
	tt = double(end - start);
	fprintf(stdout,"Using time of training:%lfs\n",tt/CLOCKS_PER_SEC);

	start = clock();
	SAFE_CALL(cudaMemcpy(filters, gpu_filters, sizeof(double) * (NEIGHBOR+1) * P_NUM * FILTERS_NUM, cudaMemcpyDeviceToHost));
	SAFE_CALL(cudaMemcpy(bias0, gpu_bias0, sizeof(double) * FILTERS_NUM, cudaMemcpyDeviceToHost));
	SAFE_CALL(cudaMemcpy(bias1, gpu_bias1, sizeof(double) * NEU_NUM1, cudaMemcpyDeviceToHost));
	SAFE_CALL(cudaMemcpy(bias2, gpu_bias2, sizeof(double) * NEU_NUM2, cudaMemcpyDeviceToHost));
	SAFE_CALL(cudaMemcpy(omega1, gpu_omega1, sizeof(double) * ome_num1, cudaMemcpyDeviceToHost));
	SAFE_CALL(cudaMemcpy(omega2, gpu_omega2, sizeof(double) * ome_num2, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	MATFile * pmatFile;
	pmatFile = matOpen("model.mat","w");
	mxArray * m1 = mxCreateDoubleMatrix((NEIGHBOR+1)*P_NUM, FILTERS_NUM,mxREAL);
	memcpy((void *)mxGetPr(m1), (void *)filters, sizeof(double) * (NEIGHBOR+1) * P_NUM * FILTERS_NUM);
	matPutVariable(pmatFile, "filters", m1);

	mxArray * m2 = mxCreateDoubleMatrix(FILTERS_NUM,1,mxREAL);
	memcpy((void *)mxGetPr(m2), (void *)bias0, sizeof(double) * FILTERS_NUM);
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
	
	//test
	double right = 0;
	double count0 = 0;
    tag = 0; // for testing
	for (int i1=0; i1<test_size; i1++){
		convolution<<<FILTERS_NUM,re_size, (NEIGHBOR+1)*z*sizeof(double)>>>(i1,gpu_processed_test,gpu_filters, gpu_re,gpu_bias0,z,re_size);
		
		maxpooling<<<FILTERS_NUM,mre_num>>>(gpu_re,gpu_mre,gpu_mre_index,re_size,mre_num);
		
		fully_connect<<<NEU_NUM1, mre_size, mre_size * sizeof(double)>>>(gpu_mre,gpu_omega1,gpu_bias1,gpu_F1,mre_size);
		
		output<<<1, NEU_NUM2>>>(tag, i1, gpu_F1, gpu_omega2, gpu_bias2, gpu_O2, gpu_processed_labels, gpu_loss);
		
		SAFE_CALL(cudaMemcpy(O2, gpu_O2, sizeof(double) * NEU_NUM2, cudaMemcpyDeviceToHost));
		
		right = count_err(test_labels, O2, i1);
		count0 = count0 + right;
	}
	end = clock();
	tt = double(end - start);
	fprintf(stdout,"Using time of testing:%lfs\n",tt/CLOCKS_PER_SEC);
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
		fprintf(stderr, "4 input arguments required!");
	}

	MATFile * datamat = matOpen(argv[1], "r");
    mxArray * train = matGetVariable(datamat,"DataSet");
    mxArray * labels = matGetVariable(datamat,"labels");
    
    trainset = (double*)mxGetData(train);
    trainlabels = (double*)mxGetData(labels);
    
    const mwSize  * dim;
    dim = mxGetDimensions(train);

    start = clock();
	double correct = training(trainset, trainlabels, dim[0], dim[1], dim[2]);
	end = clock();
    fprintf(stdout,"Correct Rate:%lf(300 iterations, train size, train:test=8:2, 0.008->0.001)\n",correct);
	double usetime = double(end - start);
	fprintf(stdout, "Using time of the whole procedure:%lfs\n",usetime/CLOCKS_PER_SEC);
	return 0;
}
