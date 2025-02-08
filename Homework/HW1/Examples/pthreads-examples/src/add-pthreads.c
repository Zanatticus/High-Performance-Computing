#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NTHREADS 10

int sum = 0;

void* adder(void* arg) {
	int             t = sum;
	struct timespec ts;
	ts.tv_sec  = 0;
	ts.tv_nsec = 100000000;   // 100 milliseconds
	nanosleep(&ts, NULL);
	sum = t + 1;
	printf("sum computed: %d\n", sum);
	return NULL;
}

int main() {
	int       i;
	pthread_t threads[NTHREADS];

	printf("forking\n");
	for (i = 0; i < NTHREADS; i++)
		if (pthread_create(threads + i, NULL, &adder, NULL) != 0)
			return i + 1;

	printf("join\n");
	for (i = 0; i < NTHREADS; i++)
		if (pthread_join(threads[i], NULL) != 0)
			return NTHREADS + i + 1;

	printf("sum computed: %d\n", sum);

	return 0;
}
