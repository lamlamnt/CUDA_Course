
////////////////////////////////////////////////////////////////////////
// GPU version of Monte Carlo algorithm using NVIDIA's CURAND library
////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>

#include <helper_cuda.h>

#define n_per_thread 100

////////////////////////////////////////////////////////////////////////
// CUDA global constants
////////////////////////////////////////////////////////////////////////
__constant__ int a = 1;
__constant__ int b = 2;
__constant__ int c = 3;

////////////////////////////////////////////////////////////////////////
// kernel routine
////////////////////////////////////////////////////////////////////////
__global__ void calc(float *output, float *input)
{
    //Version 1 (loads in contiguous memory)
    int id = threadIdx.x + n_per_thread*blockIdx.x*blockDim.x;
    //Sum up 
    float sum = 0;
    for(int i = 0; i < n_per_thread; i++)
    {
      sum += a*input[id]*input[id] + b*input[id] + c;
      id +=blockDim.x;
    }

    //Version 2 (slower version)
    /*
    int id = threadIdx.x*n_per_thread + n_per_thread*blockIdx.x*blockDim.x;
    float sum = 0;
    for(int i = 0; i < n_per_thread; i++)
    {
      sum += a*input[id]*input[id] + b*input[id] + c;
      id += 1;
    }
    */

    int output_id = threadIdx.x + blockIdx.x*blockDim.x;
    output[output_id] = (float)sum/n_per_thread;
}

////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv){
    
    //initialize variables
    float *d_input, *d_output, *h_output;
    int n_final = 1000000;
    //int n_per_thread = 100;
    int n_total = n_final*n_per_thread;
    float milli;
  // initialise card
  findCudaDevice(argc, argv);

  // initialise CUDA timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate memory on host and device

  h_output = (float *)malloc(sizeof(float)*n_final);
  checkCudaErrors( cudaMalloc((void **)&d_input, sizeof(float)*n_total) );
  checkCudaErrors( cudaMalloc((void **)&d_output, sizeof(float)*n_final) );

  // random number generation
  curandGenerator_t gen;
  checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
  checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );

  cudaEventRecord(start);
  checkCudaErrors( curandGenerateNormal(gen, d_input, n_total, 0.0f, 1.0f) );
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  printf("CURAND normal RNG  execution time (ms): %f,  samples/sec: %e \n",
          milli, n_total/(0.001*milli));

  // execute kernel and time it
  cudaEventRecord(start);

  calc<<<n_final/32, 32>>>(d_output, d_input);
  getLastCudaError("pathcalc execution failed\n");

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  printf("Monte Carlo kernel execution time (ms): %f \n",milli);

  // copy back results

  checkCudaErrors( cudaMemcpy(h_output, d_output, sizeof(float)*n_final,
                   cudaMemcpyDeviceToHost) );

  //for (int n=0; n<n_final; n++) printf(" n,  x  =  %d  %f \n",n,h_output[n]);

  //Average the contribution of each thread
  float sum1 = 0.0;
  float sum2 = 0.0;
  for (int i=0; i<n_final; i++) {
    sum1 += h_output[i];
    sum2 += h_output[i]*h_output[i];
  }

  printf("Average value of result = %.4f \n",sum1/n_final);
  printf("Standard deviation of result = %.4f \n",sum1/n_final, sqrt((sum2/n_final - (sum1/n_final)*(sum1/n_final))/n_final) );

  // Tidy up library
  checkCudaErrors( curandDestroyGenerator(gen) );

  // Release memory and exit cleanly

  free(h_output);
  checkCudaErrors( cudaFree(d_input) );
  checkCudaErrors( cudaFree(d_output) );

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

}
