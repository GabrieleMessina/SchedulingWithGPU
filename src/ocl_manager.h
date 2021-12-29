#pragma once
#include "CL/cl.h"

class OCLManager {
private:
	cl_int err;
	static OCLManager *instance;
public:
	//TODO: make kernels private
	cl_kernel entry_discover_k,
		compute_metrics_k,
		m_MergesortGlobalBigKernel,
		m_MergesortGlobalSmallKernel,
		m_MergesortStartKernel;

	cl_context ctx;
	cl_command_queue queue;
	size_t preferred_wg_size;

	static OCLManager Init(const char* progName, const char* kernelNameEntryDiscover, const char* kernelNameComputeMetrics);

	static OCLManager *GetInstance();

	cl_kernel GetEntryDiscoverKernel();
	cl_kernel GetComputeMetricsKernel();
	cl_kernel GetSortKernel(bool smallKernel = false);

	void ReleaseEntryDiscoverKernel();
	void ReleaseComputeMetricsKernel();
	void ReleaseSortKernel();
};