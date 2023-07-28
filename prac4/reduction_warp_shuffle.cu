

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

//To do block-level reduction using warp shuffling, need to store the sum of each warp in the shared memory for each block
//Because the number of warps per block is less than 32, can put all of them inside warp and do warp shuffling again BUT this is not 
//block reduction. OR can do block reduction (see which one is faster)
__global__ void reduction_warp_shuffle(float *g_odata, float *g_idata, int second_stage_len)
{
    // dynamically allocated shared memory
    extern  __shared__  float temp[];

    //tid allows to access the right element in the input array
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    int temp_indx = threadIdx.x;
    float value = g_idata[tid];
    //Perform warp shuffling sum
    for (int i=1; i<32; i*=2)
    {
        value += __shfl_xor_sync(-1, value, i);
    }
    //All value now contains the sum of the warp. Take the value of the first thread in each warp
    if (tid%32 == 0) 
    {
      temp[temp_indx/32] = value;
      //printf("%d %f \n",temp_indx/32,value);
    }
    __syncthreads();

    //Move the data from shared memory to the first warp in each block 
    float sum_of_warp;
    //Only runs for the first warp in each block -> needs to run for the first 16 threads in each block
    if(temp_indx < second_stage_len)
    {
        //Copies all 32 values into 16 slots in the first warp of every block
        sum_of_warp = temp[temp_indx];
        //printf("%d %d %f \n",tid,temp_indx,sum_of_warp);
        
        //Perform warp shuffling 
        for (int i=1; i<second_stage_len; i*=2)
        {
            sum_of_warp += __shfl_xor_sync(-1, sum_of_warp, i);
        }
    }
    //sum_of_warp now contains the sum of one block
    //Why don't need sync threads here!!!!
    // finally, first thread puts result into global memory
    if (temp_indx==0) 
    {
      //printf("%f \n",sum_of_warp);
      g_odata[blockIdx.x] = sum_of_warp;
    }
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
  //shared_mem_size = sizeof(float) * num_threads;
  int second_stage_len = num_threads/32;
  shared_mem_size = sizeof(float) * second_stage_len;
  //shared_mem_size = sizeof(float) * num_threads;
  // initialise CUDA timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  reduction_warp_shuffle<<<num_blocks,num_threads,shared_mem_size>>>(d_odata,d_idata,second_stage_len);
  cudaEventRecord(stop);
  getLastCudaError("reduction kernel execution failed");
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  printf("Execution time using warp shuffling sum (ms): %f \n",milli);

  
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
