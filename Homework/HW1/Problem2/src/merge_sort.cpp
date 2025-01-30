// CPP Program to implement merge sort using multi-threading
// Code taken from:
// https://www.geeksforgeeks.org/merge-sort-using-multi-threading/ Code
// modified by: Zander Ingare

#include <iostream>
#include <pthread.h>
#include <time.h>

// number of elements in array
#define MAX 10000

// number of threads
#define THREAD_MAX 32

using namespace std;

// array of size MAX
int a[MAX];
int part = 0;

// merge function for merging two parts
void merge(int low, int mid, int high) {
  int *left = new int[mid - low + 1];
  int *right = new int[high - mid];

  // n1 is size of left part and n2 is size of right part
  int n1 = mid - low + 1, n2 = high - mid, i, j;

  // storing values in left part
  for (i = 0; i < n1; i++)
    left[i] = a[i + low];

  // storing values in right part
  for (i = 0; i < n2; i++)
    right[i] = a[i + mid + 1];

  int k = low;
  i = j = 0;

  // merge left and right in ascending order
  while (i < n1 && j < n2) {
    if (left[i] <= right[j])
      a[k++] = left[i++];
    else
      a[k++] = right[j++];
  }

  // insert remaining values from left
  while (i < n1) {
    a[k++] = left[i++];
  }

  // insert remaining values from right
  while (j < n2) {
    a[k++] = right[j++];
  }
}

// merge sort function
void merge_sort(int low, int high) {
  // calculating mid point of array
  int mid = low + (high - low) / 2;
  if (low < high) {
    // calling first half
    merge_sort(low, mid);

    // calling second half
    merge_sort(mid + 1, high);

    // merging the two halves
    merge(low, mid, high);
  }
}

// thread function for multi-threading
void *merge_sort_thread(void *arg) {
  int thread_part = *((int *)arg);

  // calculating low and high
  int low = thread_part * (MAX / THREAD_MAX);
  int high = (thread_part + 1) * (MAX / THREAD_MAX) - 1;

  // evaluating mid point
  int mid = low + (high - low) / 2;
  if (low < high) {
    merge_sort(low, mid);
    merge_sort(mid + 1, high);
    merge(low, mid, high);
  }
  return NULL;
}

// Driver Code
int main() {
  // generating random values in array
  for (int i = 0; i < MAX; i++)
    a[i] = rand() % 100;

  // t1 and t2 for calculating time for merge sort
  clock_t t1, t2;

  t1 = clock();
  pthread_t threads[THREAD_MAX];
  int thread_part[THREAD_MAX];

  // creating threads
  for (int i = 0; i < THREAD_MAX; i++) {
    thread_part[i] = i;
    pthread_create(&threads[i], NULL, merge_sort_thread,
                   (void *)&thread_part[i]);
  }

  // joining all threads
  for (int i = 0; i < THREAD_MAX; i++)
    pthread_join(threads[i], NULL);

  // merging the final parts
  int current_size = MAX / THREAD_MAX;
  for (int size = current_size; size < MAX; size *= 2) {
    for (int low = 0; low < MAX - size; low += 2 * size) {
      int mid = low + size - 1;
      int high = min(low + 2 * size - 1, MAX - 1);
      merge(low, mid, high);
    }
  }

  t2 = clock();

  // displaying sorted array
  cout << "Sorted array: ";
  for (int i = 0; i < MAX; i++)
    cout << a[i] << " ";

  // time taken by merge sort in miliseconds
  cout << "\n\nSuccessfully Merge Sorted!\nTime taken: "
       << 1000*(t2 - t1) / (double)CLOCKS_PER_SEC << " ms" << endl;

  return 0;
}
