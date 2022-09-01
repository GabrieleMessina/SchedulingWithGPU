#pragma once
#include "ocl_boiler.h"
#include "app_globals.h"
#include <tuple>

#define MERGESORT_SMALL_STRIDE 1024

class SortMetrics {
private:
	static cl_event run_sort_kernel(int metrics_len, int stride, bool smallKernel, bool flip);
public:
	static std::tuple<cl_event*, metrics_t*> MergeSort(metrics_t* metrics, int n_nodes);
};