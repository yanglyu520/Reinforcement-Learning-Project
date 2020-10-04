#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <CL/cl.h>
#include <mpi.h>
using namespace std;

#define MASTER 0
#define MASTER_TAG 1
#define WORKER_TAG 2
const int N = 10;
const int MAX_VALUE = 1000;
int v[N], test[N];
const int mid = N / 2;

cl_device_id device_id;
cl_context context;
cl_program program;
cl_kernel kernel;
cl_command_queue queue;

cl_event event = NULL;
int err;

int left_half[mid], right_half[N-mid];
cl_mem buff, buffRes;
const int n = mid;
const size_t local[1] = { n };
const size_t global[1] = { n };
void free_memory();
void copy_kernel_args();
void setup_kernel_memory(int left_half[]);
void setup_openCL_device_context_queue_kernel();
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename);
cl_device_id create_device();

void free_memory() {
  clReleaseKernel(kernel);
  clReleaseMemObject(buff);
  clReleaseMemObject(buffRes);

  clReleaseCommandQueue(queue);
  clReleaseProgram(program);
  clReleaseContext(context);
}

void copy_kernel_args() {
  clSetKernelArg(kernel, 0, sizeof(int), (void*)&n);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buff);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buffRes);
  if(err < 0) {
    perror("Couldn't create a kernel argument");
    printf("error = %d", err);
    exit(1);
  }
}

void setup_kernel_memory(int left_half[]) {
  buff = clCreateBuffer(context, CL_MEM_READ_WRITE,  n*sizeof(int), NULL, NULL);
  buffRes = clCreateBuffer(context, CL_MEM_READ_WRITE,  n*sizeof(int), NULL, NULL);

  // Copy matrices to the GPU
  clEnqueueWriteBuffer(queue, buff, CL_TRUE, 0, n*sizeof(int), left_half, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, buffRes, CL_TRUE, 0, n*sizeof(int), left_half, 0, NULL, NULL);
}

void setup_openCL_device_context_queue_kernel() {
  device_id = create_device();
  cl_int err;
  context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
  if(err < 0) {
    perror("Couldn't create a context");
    exit(1);   
  }

  program = build_program(context, device_id, "quick_sort.cl");

  queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
  if(err < 0) {
    perror("Couldn't create a command queue");
    exit(1);   
  };

  kernel = clCreateKernel(program, "quick_sort", &err);
  if(err < 0) {
    perror("Couldn't create a kernel");
    printf("error =%d", err);
    exit(1);
  };
}

cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {
  cl_program program;
  FILE *program_handle;
  char *program_buffer, *program_log;
  size_t program_size, log_size;

  program_handle = fopen(filename, "r");
  if(program_handle == NULL) {
    perror("Couldn't find the program file");
    exit(1);
  }

  fseek(program_handle, 0, SEEK_END);
  program_size = ftell(program_handle);
  rewind(program_handle);
  program_buffer = (char*)malloc(program_size + 1);
  program_buffer[program_size] = '\0';
  fread(program_buffer, sizeof(char), program_size, program_handle);
  fclose(program_handle);

  program = clCreateProgramWithSource(ctx, 1, 
      (const char**)&program_buffer, &program_size, &err);
  if(err < 0) {
    perror("Couldn't create the program");
    exit(1);
  }
  free(program_buffer);

  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if(err < 0) {

    /* Find size of log and print to std output */
    clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
        0, NULL, &log_size);
    program_log = (char*) malloc(log_size + 1);
    program_log[log_size] = '\0';
    clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
        log_size + 1, program_log, NULL);
    printf("%s\n", program_log);
    free(program_log);
    exit(1);
  }

  return program;
}

cl_device_id create_device() {

  cl_platform_id platform;
  cl_device_id dev;
  int err;

  err = clGetPlatformIDs(1, &platform, NULL);
  if(err < 0) {
    perror("Couldn't identify a platform");
    exit(1);
  } 

  // GPU
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
  if(err == CL_DEVICE_NOT_FOUND) {
    // CPU
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
  }
  if(err < 0) {
    perror("Couldn't access any devices");
    exit(1);   
  }

  return dev;
}

bool is_sorted(int source[], int target[]) {
  sort(source, source+N);
  for(int i = 0; i < N; i++)
    if(source[i] != target[i])
      return false;
  return true;
}

int partition(int v[], int start, int end) {
  int pivotValue = v[end];
  int pivotIndex = start;

  for(int i = start; i < end; i++) {
    if(v[i] < pivotValue) {
      swap(v[i], v[pivotIndex]);
      pivotIndex++;
    }
  }

  swap(v[pivotIndex], v[end]);
  return pivotIndex;
}
void quick_sort(int v[], int start, int end) {
  if(start >= end)
    return;

  int index = partition(v, start, end);

  quick_sort(v, start, index - 1);
  quick_sort(v, index + 1, end);
}

int main(int argc, char *argv[]) {
  int num_process, process_id;
  MPI_Status status;
  
  // Seed rand function
  srand(time(0));

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
  MPI_Comm_size(MPI_COMM_WORLD, &num_process);

  
  /* 
   * MASTER Process
   */
  if(process_id == MASTER) {
    for(int i = 0; i < N; i++) {
      v[i] = rand() % MAX_VALUE;
      test[i] = v[i];
      if(i < mid)
        left_half[i] = v[i];
      else
        right_half[i-mid] = v[i]; 
    }
    
    
    // Start measuring execution time of multiplication operation
    clock_t start = clock();

    // Start QuickSort
    // Send left half to worker and keep right half for master
    printf("\n%d leftmost elements are sent to node worker", mid);
    MPI_Send(&left_half, mid, MPI_INT, 1, MASTER_TAG, MPI_COMM_WORLD);
    
    // Sort right half
    quick_sort(right_half, 0, N - mid - 1);

    // Received sorted left half from node worker
    printf("\nReceived %d elements from node worker", mid); 
    MPI_Recv(&left_half, mid, MPI_INT, 1, WORKER_TAG, MPI_COMM_WORLD, &status);

    // Stop measuring execution time of multiplication operation
    clock_t end = clock();
    // Print execution time
    double execution_time = double(end-start) / (CLOCKS_PER_SEC / 1000);
    cout << "\n\nTime of gaining the dot product of two above matrices: " << execution_time << "ms\n"; 


    // Merge two halves into one
    int left = 0, right = 0, k = 0;
    while((left < mid) && (right < N - mid)) {
      if(left_half[left] < right_half[right])
        v[k++] = left_half[left++];
      else
        v[k++] = right_half[right++];
    }
    while(left < mid)
      v[k++] = left_half[left++];
    while(right < N - mid)
      v[k++] = right_half[right++];

    cout << "Left Half: ";
    for(int i = 0; i < mid; i++)
      cout << left_half[i] << ' ';
    cout << '\n';
    cout << "Right half: ";
    for(int i = 0; i < N-mid; i++)
      cout << right_half[i] << ' ';
    cout << '\n';
    cout << "The find sorted array: ";
    for(int i = 0; i < N; i++)
      cout << v[i] << ' ';
    cout << '\n';

    if(is_sorted(test, v))
      cout << "The vector is sorted";
    else 
      cout << "The vector is not sorted";
    cout << '\n';
  }

  /* 
   * Node Worker
   */

  if(process_id != MASTER) {
    // Receive left half from master
    MPI_Recv(&left_half, mid, MPI_INT, MASTER, MASTER_TAG, MPI_COMM_WORLD, &status);
    printf("\nReceived %d elements from master", mid); 

    // Sort left half
    setup_openCL_device_context_queue_kernel();
    setup_kernel_memory(left_half);
    copy_kernel_args();
    clWaitForEvents(1, &event);

    clEnqueueReadBuffer(queue, buffRes, CL_TRUE, 0, n*sizeof(int), left_half, 0, NULL, NULL);
    free_memory();

    quick_sort(left_half, 0, mid - 1);
    // Send sorted array back to master
    printf("\n%d leftmost elements are sent back to master\n", mid);
    MPI_Send(&left_half, mid, MPI_INT, MASTER, WORKER_TAG, MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return 0; 
}
