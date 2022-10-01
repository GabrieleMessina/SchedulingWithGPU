#include "sort_metrics.h"
#include "app_globals.h"
#include "ocl_manager.h"
#include "ocl_buffer_manager.h"
#include <iostream>
#include <math.h>
#include "utils.h"


tuple<cl_event*, metrics_t*> SortMetrics::BitonicMergesort(metrics_t* metrics, int n_nodes)
{
	OCLBufferManager BufferManager = *OCLBufferManager::GetInstance();
	cl_event sort_task_evts_start = cl_event();
	cl_event sort_task_evts_end = cl_event();

	const int metrics_len = GetMetricsArrayLenght(n_nodes);
	metrics_t* ordered_metrics = DBG_NEW metrics_t[metrics_len]; for (int i = 0; i < metrics_len; i++) ordered_metrics[i] = metrics[i];
	BufferManager.SetOrderedMetrics(metrics);

	unsigned int limit = (unsigned int)2 * OCLManager::preferred_wg_size; //limit is double the localWorkSize
	bool flip = false;

	// start with Sort_BitonicMergesortLocalBegin to sort local until we reach the limit
	sort_task_evts_start = sort_task_evts_end = SortMetrics::run_bitonic_sort_kernel(metrics_len, NULL, NULL, 0, flip);

	// proceed with global and local kernels
	for (unsigned int blocksize = limit; blocksize <= metrics_len; blocksize <<= 1) {
		for (unsigned int stride = blocksize / 2; stride > 0; stride >>= 1) {
			if (stride >= limit) {
				//Sort_BitonicMergesortGlobal
				sort_task_evts_end = SortMetrics::run_bitonic_sort_kernel(metrics_len, stride, blocksize, 2, !flip);
			}
			else {
				//Sort_BitonicMergesortLocal
				sort_task_evts_end = SortMetrics::run_bitonic_sort_kernel(metrics_len, stride, blocksize, 1, !flip);
			}
		}
	}
	flip = !flip; //non ha alcuno scopo, l'algoritmo originale lo aveva e l'ho mantenuto ma la struttura flip-flop non viene usata.

	BufferManager.GetOrderedMetricsResult(ordered_metrics, &sort_task_evts_end, 1);


	if (DEBUG_SORT) {
		cout << "metrics sorted: (rank, level, task_id)" << endl;
		print(ordered_metrics, metrics_len, "\n", true, 0);
		//print(ordered_metrics, metrics_len, "\n"); //per vedere anche i dati aggiunti per padding
		cout << "\n";
	}
	else if (DEBUG_SORT_PARTIAL) { //mostro solo i primi e gli ultimi 5 elementi per essere sicuro che tutto abbia funzionato.
		cout << "metrics sorted: (rank, level, task_id)" << endl;
		print(ordered_metrics, min(metrics_len, 5), "\n", true, 0);
		cout << "[...]" << endl << endl;
		print(ordered_metrics, metrics_len, "\n", true, metrics_len - 5);
		cout << "\n";
	}
	else {
		//cout << endl << "metrics sorted" << endl;
	}

	BufferManager.ReleaseMetrics();
	BufferManager.ReleaseOrderedMetrics();

	cl_event* sort_task_evts = DBG_NEW cl_event[2];
	sort_task_evts[0] = sort_task_evts_start;
	sort_task_evts[1] = sort_task_evts_end;
	return make_tuple(sort_task_evts, ordered_metrics);
}

tuple<cl_event*, metrics_t*> SortMetrics::MergeSort(metrics_t* metrics, int n_nodes) {
	const int metrics_len = GetMetricsArrayLenght(n_nodes);
	OCLBufferManager BufferManager= *OCLBufferManager::GetInstance();

	metrics_t* ordered_metrics = DBG_NEW metrics_t[metrics_len]; for (int i = 0; i < metrics_len; i++) ordered_metrics[i] = metrics[i];
	BufferManager.SetOrderedMetrics(metrics);

	unsigned int locLimit = 1;
	unsigned int stride = 2 * locLimit;
	bool useSmallKernel = metrics_len <= MERGESORT_SMALL_STRIDE;

	bool flip = false;
	int task_event_launched = 0;

	cl_event sort_task_evts_start = cl_event();
	cl_event sort_task_evts_end = cl_event();
	//cout << "small kernel? " << (useSmallKernel ? "true" : "false") << ", metrics len: " << (int)metrics_len << endl;
	for (; stride <= metrics_len; stride <<= 1 /*stride x 2*/) { // crea i branch per il merge sort.
		//calculate work sizes
		sort_task_evts_end = run_sort_kernel(metrics_len, stride, useSmallKernel, flip);
		if (task_event_launched == 0) sort_task_evts_start = sort_task_evts_end;

		if (!useSmallKernel && stride >= 1024 * 1024) ocl_check(clFinish(OCLManager::queue), "Failed finish CommandQueue at mergesort for bigger strides.");

		flip = !flip;
		task_event_launched++;
	}

	if(isEven(task_event_launched))
		BufferManager.GetMetricsResult(ordered_metrics, &sort_task_evts_end, 1);//turns out this is the right array to read from because the other one is missing the last step of course...
	else 
		BufferManager.GetOrderedMetricsResult(ordered_metrics, &sort_task_evts_end, 1);

	if (DEBUG_SORT) {
		cout << "metrics sorted: (rank, level, task_id)" << endl;
		cout << "sort kernel launch count:" << task_event_launched << endl;
		print(ordered_metrics, metrics_len, "\n", true, 0);
		//print(ordered_metrics, metrics_len, "\n"); //per vedere anche i dati aggiunti per padding
		cout << "\n";
	}
	else if(DEBUG_SORT_PARTIAL){ //mostro solo i primi e gli ultimi 5 elementi per essere sicuro che tutto abbia funzionato.
		cout << "metrics sorted: (rank, level, task_id)" << endl;
		print(ordered_metrics, min(metrics_len, 5), "\n", true, 0);
		cout << "[...]" << endl << endl;
		print(ordered_metrics, metrics_len, "\n", true, metrics_len - 5);
		cout << "\n";
	}
	else {
		//cout << endl << "metrics sorted" << endl;
	}

	BufferManager.ReleaseMetrics();
	BufferManager.ReleaseOrderedMetrics();

	cl_event* sort_task_evts = DBG_NEW cl_event[2];
	sort_task_evts[0] = sort_task_evts_start;
	sort_task_evts[1] = sort_task_evts_end;
	return make_tuple(sort_task_evts, ordered_metrics);
}

cl_event SortMetrics::run_bitonic_sort_kernel(int metrics_len, int stride, int blocksize, int startSmallBigKernel, bool flip) {
	OCLBufferManager BufferManager = *OCLBufferManager::GetInstance();

	size_t lws[] = { OCLManager::preferred_wg_size};
	size_t gws[] = { GetGlobalWorkSize(metrics_len / 2, lws[0]) };


	cl_mem metrics_GPU, ordered_metrics_GPU;
	if (flip) {
		metrics_GPU = BufferManager.GetOrderedMetrics();
		ordered_metrics_GPU = BufferManager.GetMetrics();
	}
	else {
		metrics_GPU = BufferManager.GetMetrics();
		ordered_metrics_GPU = BufferManager.GetOrderedMetrics();
	}

	cl_int err;
	int arg_index = 0;

	if (startSmallBigKernel > 0) {
		err = clSetKernelArg(OCLManager::GetBitonicSortKernel(startSmallBigKernel), arg_index++, sizeof(metrics_GPU), &metrics_GPU);
		ocl_check(err, "set arg %d for sort kernel", arg_index);
		err = clSetKernelArg(OCLManager::GetBitonicSortKernel(startSmallBigKernel), arg_index++, sizeof(cl_uint), (void*)&metrics_len);
		ocl_check(err, "set arg %d for sort kernel", arg_index);
		err = clSetKernelArg(OCLManager::GetBitonicSortKernel(startSmallBigKernel), arg_index++, sizeof(cl_uint), (void*)&blocksize);
		ocl_check(err, "set arg %d for sort kernel", arg_index);
		err = clSetKernelArg(OCLManager::GetBitonicSortKernel(startSmallBigKernel), arg_index++, sizeof(cl_uint), (void*)&stride);
		ocl_check(err, "set arg %d for sort kernel", arg_index);
	}
	else {
		err = clSetKernelArg(OCLManager::GetBitonicSortKernel(startSmallBigKernel), arg_index++, sizeof(metrics_GPU), &metrics_GPU);
		ocl_check(err, "set arg %d for sort kernel", arg_index);
		err = clSetKernelArg(OCLManager::GetBitonicSortKernel(startSmallBigKernel), arg_index++, sizeof(ordered_metrics_GPU), &ordered_metrics_GPU);
		ocl_check(err, "set arg %d for sort kernel", arg_index);
	}

	cl_event ordered_metrics_evt;
	err = clEnqueueNDRangeKernel(OCLManager::queue,
		OCLManager::GetBitonicSortKernel(startSmallBigKernel),
		1, NULL, gws, lws,
		0, NULL, &ordered_metrics_evt);

	ocl_check(err, "enqueue bitonic sort kernel");

	return ordered_metrics_evt;
}

cl_event SortMetrics::run_sort_kernel(int metrics_len, int stride, bool smallKernel, bool flip) {
	OCLBufferManager BufferManager = *OCLBufferManager::GetInstance();

	/*size_t lws[] = { OCLManager::preferred_wg_size };
	size_t gws[] = { GetGlobalWorkSize(metrics_len / 2, lws[0]) };*/
	size_t neededWorkers = metrics_len / stride;
	size_t lws[] = { min(OCLManager::preferred_wg_size, neededWorkers) };
	size_t gws[] = { GetGlobalWorkSize(neededWorkers, lws[0]) };


	cl_mem metrics_GPU, ordered_metrics_GPU;
	if (flip) {
		metrics_GPU = BufferManager.GetOrderedMetrics();
		ordered_metrics_GPU = BufferManager.GetMetrics();
	}
	else {
		metrics_GPU = BufferManager.GetMetrics();
		ordered_metrics_GPU = BufferManager.GetOrderedMetrics();
	}

	cl_int err;
	int arg_index = 0;

	err = clSetKernelArg(OCLManager::GetSortKernel(smallKernel), arg_index++, sizeof(metrics_GPU), &metrics_GPU);
	ocl_check(err, "set arg %d for sort kernel", arg_index); 
	err = clSetKernelArg(OCLManager::GetSortKernel(smallKernel), arg_index++, sizeof(ordered_metrics_GPU), &ordered_metrics_GPU);
	ocl_check(err, "set arg %d for sort kernel", arg_index);
	err = clSetKernelArg(OCLManager::GetSortKernel(smallKernel), arg_index++, sizeof(cl_uint), (void*)&stride);
	ocl_check(err, "set arg %d for sort kernel", arg_index);
	err = clSetKernelArg(OCLManager::GetSortKernel(smallKernel), arg_index++, sizeof(cl_uint), (void*)&metrics_len);
	ocl_check(err, "set arg %d for sort kernel", arg_index);

	cl_event ordered_metrics_evt;
	err = clEnqueueNDRangeKernel(OCLManager::queue,
		OCLManager::GetSortKernel(smallKernel),
		1, NULL, gws, lws,
		0, NULL, &ordered_metrics_evt);

	ocl_check(err, "enqueue sort kernel");

	return ordered_metrics_evt;
}