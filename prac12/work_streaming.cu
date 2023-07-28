#include <stdio.h>
#include <cuda.h>

#include "helper_cuda.h"

__global__ void do_work(double *data, int N, int idx, int chunk_num) {
	printf("Start doing computing %d \n", chunk_num);
	int i = blockIdx.x * blockDim.x + blockDim.x*idx + threadIdx.x;
	if (i < N) {
		for (int j = 0; j < 20; j++) {
			data[i] = cos(data[i]);
			data[i] = sqrt(fabs(data[i]));
		}
	}
}

int main()
{
	//Allocate 1 GB of data
	int total_data = 1<<27;
	//double *d_data;
	double *h_data;
	int stream_number = 3; //One stream that copies host to device, one that does computation, and one that copies device to host
	int num_chunks = 10;
	h_data = (double*)malloc(total_data*sizeof(double));
	//checkCudaErrors(cudaMalloc( (void**)&d_data, total_data*sizeof(double) ));

	//Initialise host data
	srand(0);
	for (int i = 0; i < total_data; i++)
		h_data[i] = (double)rand()/(double)RAND_MAX;

	//Start timing	
	float time;
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));

	//Divide the host data up to multiple parts 
	double *h_data_list[num_chunks]; 
	double *d_data_list[num_chunks];
	//Assume it's divisible!!!
	int num_elements_per_chunk = (int)total_data/num_chunks;
	for (int i = 0; i < num_chunks; ++i) 
	{
        h_data_list[i] = &h_data[i*num_elements_per_chunk];
    }

	//Stream that copies the data from host to device
	cudaStream_t stream_copyhd;
	cudaStreamCreate(&stream_copyhd);
	//Use pinned memory on the host
	for(int i = 0; i < num_chunks; ++i)
	{
		//allocates fixed memory to the CPU
		printf("Start copying chunk of data %d from host to device \n",i);
		cudaError_t status = cudaMallocHost((void**)&h_data_list[i], sizeof(double)*num_elements_per_chunk);
		if (status != cudaSuccess)
  			printf("Error allocating pinned host memory\n");
		checkCudaErrors(cudaMalloc( (void**)&d_data_list[i], num_elements_per_chunk*sizeof(double) ));
		cudaMemcpyAsync(d_data_list[i],h_data_list[i],num_elements_per_chunk*sizeof(double),cudaMemcpyHostToDevice,stream_copyhd);
	}

	//Stream that does the computation
	cudaStream_t stream_compute;
	cudaStreamCreate(&stream_compute);
	int blocksize_stream = 256;
	int nblocks_stream = (num_elements_per_chunk-1)/blocksize_stream + 1;
	for(int i = 0; i < num_chunks; ++i)
	{
		printf("Launch kernel for chunk %d \n",i);
		do_work<<<blocksize_stream,nblocks_stream,0,stream_compute>>>(d_data_list[i], num_elements_per_chunk, nblocks_stream, i);
	}

	cudaStream_t stream_copydh;
	cudaStreamCreate(&stream_copydh);
	for(int i = 0; i < num_chunks;++i)
	{
		printf("Start copying chunk of data %d from device to host \n",i);
		cudaMemcpyAsync(h_data_list[i],d_data_list[i],num_elements_per_chunk*sizeof(double),cudaMemcpyDeviceToHost,stream_copydh);
	}

	/*
	//Copy data to device
	checkCudaErrors(cudaMemcpy(d_data,h_data,total_data*sizeof(double),cudaMemcpyHostToDevice));

	//Figure out how many blocks are needed
	int blocksize = 256;
	int nblocks = (total_data-1)/blocksize+1;

	//Launch kernel to process data
	do_work<<<nblocks,blocksize,0,0>>>(d_data, total_data, 0*nblocks);

	//Copy data back from device
	checkCudaErrors(cudaMemcpy(h_data,d_data,total_data*sizeof(double),cudaMemcpyDeviceToHost));
	*/
	cudaDeviceSynchronize();
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
	printf("Total processing time:  %g ms\n", time);

	//checkCudaErrors(cudaFree( d_data ));
	free(h_data);
	for (int i = 0; i < num_chunks; ++i) 
	{
		checkCudaErrors(cudaFreeHost(h_data_list[i]));
		checkCudaErrors(cudaFree(d_data_list[i]));
	}
	cudaDeviceReset();
	return EXIT_SUCCESS;
}

