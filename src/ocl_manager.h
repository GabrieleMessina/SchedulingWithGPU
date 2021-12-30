#pragma once
#include "CL/cl.h"

class OCLManager {
private:
	static cl_int err;
	static cl_program prog;
	static cl_program prog2;
	static cl_kernel entry_discover_k,
		compute_metrics_k,
		m_MergesortGlobalBigKernel,
		m_MergesortGlobalSmallKernel,
		m_MergesortStartKernel;
public:
	static cl_context ctx;
	static cl_command_queue queue;
	static size_t preferred_wg_size;

	
	static void Init(const char* progName, const char* kernelNameEntryDiscover, const char* kernelNameComputeMetrics);
	static void Reset();
	static void Release();

	static cl_kernel GetEntryDiscoverKernel();
	static cl_kernel GetComputeMetricsKernel();
	static cl_kernel GetSortKernel(bool smallKernel = false);

	static void ReleaseEntryDiscoverKernel();
	static void ReleaseComputeMetricsKernel();
	static void ReleaseSortKernel();
};