//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <helper_cuda.h>


//
// kernel routine
// 

__global__ void my_first_kernel(float *x, float *input1, float *input2)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  //x[tid] = (float) threadIdx.x;
  x[tid] = (float) (input1[tid] + input2[tid]);
}

//
// main code
//

int main(int argc, const char **argv)
{
  float *h_x, *d_x;
  int   nblocks, nthreads, nsize, n; 
  float *h_input1, *h_input2, *d_input1, *d_input2;

  // initialise card

  findCudaDevice(argc, argv);

  // set number of blocks, and threads per block
  nblocks  = 2;
  nthreads = 32;
  nsize    = nblocks*nthreads ;

  // allocate memory for array

  h_x = (float *)malloc(nsize*sizeof(float));
  checkCudaErrors(cudaMalloc((void **)&d_x, nsize*sizeof(float)));

  h_input1 = (float *)malloc(nsize*sizeof(float));
  h_input2 = (float *)malloc(nsize*sizeof(float));
  checkCudaErrors(cudaMalloc((void **)&d_input1, nsize*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_input2, nsize*sizeof(float)));

  //Initialize numbers for h_input1 and h_input2 OR alternatively, initialize in kernel
  for (int i = 0; i < nsize; i++) 
  {
    h_input1[i] = (float) i; // Assigning some sample values (you can change this as well)
    h_input2[i] = (float) i;
  }

  //Copy from h to d
  checkCudaErrors(cudaMemcpy(d_input1,h_input1,nsize*sizeof(float),
                 cudaMemcpyHostToDevice) );
  checkCudaErrors(cudaMemcpy(d_input2,h_input2,nsize*sizeof(float),
                 cudaMemcpyHostToDevice) );
  
  // execute kernel
  
  my_first_kernel<<<nblocks,nthreads>>>(d_x, d_input1, d_input2);
  getLastCudaError("my_first_kernel execution failed\n");

  // copy back results and print them out

  checkCudaErrors( cudaMemcpy(h_x,d_x,nsize*sizeof(float),
                 cudaMemcpyDeviceToHost) );

  for (n=0; n<nsize; n++) printf(" n,  x  =  %d  %f \n",n,h_x[n]);

  // free memory 

  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_input1));
  checkCudaErrors(cudaFree(d_input2));
  free(h_x);
  free(h_input1);
  free(h_input2);

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}
