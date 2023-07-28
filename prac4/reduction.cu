

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////////////
// CPU routines
////////////////////////////////////////////////////////////////////////////////

float reduction_gold(float* idata, int len) 
{
  float sum = 0.0f;
  for(int i=0; i<len; i++) sum += idata[i];

  return sum;
}

////////////////////////////////////////////////////////////////////////////////
// GPU routines
////////////////////////////////////////////////////////////////////////////////

//Performs reduction sum within 1 block
//Each block works with a different section of the array
//Shared memory is only shared in each block
__global__ void reduction(float *g_odata, float *g_idata)
{
    // dynamically allocated shared memory
    extern  __shared__  float temp[];

    //tid allows to access the right element in the input array
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    int temp_indx = threadIdx.x;

    // first, each thread loads data into shared memory
    temp[temp_indx] = g_idata[tid];

    // next, we perform binary tree reduction

    for (int d=blockDim.x/2; d>0; d=d/2) {
      __syncthreads();  // ensure previous step completed 
      if (temp_indx<d)  temp[temp_indx] += temp[temp_indx+d];
    }

    // finally, first thread puts result into global memory
    if (temp_indx==0) g_odata[blockIdx.x] = temp[0];

    //Do atomic add - can only do atomic add for 2 values at a time
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main( int argc, const char** argv) 
{
  int num_blocks, num_elements, num_threads, mem_size, shared_mem_size;

  float *h_data, sum;
  float *d_idata, *d_odata;
  float *h_out;
  float sum_gpu;
  float milli;
  //d_odata contains the partial sums. Num of elements equals num_blocks

  // initialise card

  findCudaDevice(argc, argv);

  num_blocks   = 30;  // start with only 1 thread block
  num_threads = 512;
  num_elements = num_blocks*num_threads;
  mem_size     = sizeof(float) * num_elements;

  // allocate host memory to store the input data
  // and initialize to integer values between 0 and 10

  h_data = (float*) malloc(mem_size);
  h_out = (float*)malloc(sizeof(float)*num_blocks);
      
  for(int i = 0; i < num_elements; i++) 
    h_data[i] = floorf(10.0f*(rand()/(float)RAND_MAX));

  // compute reference solution

  sum = reduction_gold(h_data, num_elements);

  // allocate device memory input and output arrays

  checkCudaErrors( cudaMalloc((void**)&d_idata, mem_size) );
  checkCudaErrors( cudaMalloc((void**)&d_odata, sizeof(float)*num_blocks) );

  // copy host memory to device input array

  checkCudaErrors( cudaMemcpy(d_idata, h_data, mem_size,
                              cudaMemcpyHostToDevice) );

  // execute the kernel

  shared_mem_size = sizeof(float) * num_threads;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  reduction<<<num_blocks,num_threads,shared_mem_size>>>(d_odata,d_idata);
  getLastCudaError("reduction kernel execution failed");
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  printf("Execution time using reduction (ms): %f \n",milli);

  // copy result from device to host

  checkCudaErrors( cudaMemcpy(h_out, d_odata, sizeof(float)*num_blocks,
                              cudaMemcpyDeviceToHost) );

  // check results
  //print the partial sums
  //for (int n=0; n<num_blocks; n++) printf("%f \n",h_out[n]);

  //Sum up partial sums in the host
  for(int i = 0; i<num_blocks;i++)
  {
    sum_gpu += h_out[i];
  }
  printf("The sum is %f \n",sum_gpu);
  printf("reduction error = %f\n",sum_gpu-sum);

  // cleanup memory
  
  checkCudaErrors( cudaFree(d_idata) );
  checkCudaErrors( cudaFree(d_odata) );
  
  free(h_data);
  free(h_out);

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();
}
