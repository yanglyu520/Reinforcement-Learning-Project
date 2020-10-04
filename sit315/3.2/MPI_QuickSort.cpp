#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <mpi.h>
using namespace std;

#define MASTER 0
#define MASTER_TAG 1
#define WORKER_TAG 2
const int N = 5000000;
const int MAX_VALUE = 1000;
int v[N], test[N];
const int mid = N / 2;
int left_half[mid], right_half[N-mid];

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

    // cout << "Left Half: ";
    // for(int i = 0; i < mid; i++)
    //   cout << left_half[i] << ' ';
    // cout << '\n';
    // cout << "Right half: ";
    // for(int i = 0; i < N-mid; i++)
    //   cout << right_half[i] << ' ';
    // cout << '\n';
    // cout << "The find sorted array: ";
    // for(int i = 0; i < N; i++)
    //   cout << v[i] << ' ';
    // cout << '\n';

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
    quick_sort(left_half, 0, mid - 1);

    // Send sorted array back to master
    printf("\n%d leftmost elements are sent back to master\n", mid);
    MPI_Send(&left_half, mid, MPI_INT, MASTER, WORKER_TAG, MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return 0; 
}
