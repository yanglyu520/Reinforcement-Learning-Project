int partition(__global int *arr, int l, int h) 
{ 
    int x = arr[h]; 
    int i = (l - 1);
    int temp;
  
    for (int j = l; j <= h - 1; j++) { 
        if (arr[j] <= x) { 
            i++;
            temp = arr[i];
            arr[i] = arr[j];
	    arr[j] = temp;  
        } 
    }
    temp = arr[i+1];
    arr[i+1] = arr[h];
    arr[h] = temp; 
    return (i + 1); 
}
__kernel void raindomized_quick_sort(__global int* arr, __global int *stack, int low, int high) {
  
   int top = -1; 

   stack[++top] = low; 
   stack[++top] = high; 
   
   while (top >= 0) { 
     high = stack[top--]; 
     low = stack[top--]; 
     int p = partition(arr, low, high); 
  
     if (p - 1 > low) { 
        stack[++top] = low; 
        stack[++top] = p - 1; 
      } 
  

      if (p + 1 < high) { 
         stack[++top] = p + 1; 
         stack[++top] = high; 
      } 
   } 
}

__kernel void quick_sort(const int n, __global int* arr, __global int* arrRes) {
    
    // Thread identifiers
    const int quantity = get_global_id(0);
    raindomized_quick_sort(arr, arrRes, 0, quantity - 1);

    for(int i = 0; i < quantity; i++)
	arrRes[i] = arr[i];
}



