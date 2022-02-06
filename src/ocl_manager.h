#pragma once
#include "CL/cl.h"

enum class ComputeMetricsVersion { Latest, Working, v1, v2 };
enum class VectorizedComputeMetricsVersion { Latest, Working, v1, v2 };

class OCLManager {
private:
	static cl_int err;
	static cl_program entry_discover_prog;
	static cl_program compute_metrics_prog;
	static cl_program sort_prog;
	static cl_kernel entry_discover_k,
		compute_metrics_k,
		m_MergesortGlobalBigKernel,
		m_MergesortGlobalSmallKernel,
		m_MergesortStartKernel;
	static void InitCommon(const char* entryDiscoverKernelName, const char* computeMetricsKernelName, const char* sortKernelName);
public:
	static ComputeMetricsVersion compute_metrics_version_chosen;
	static VectorizedComputeMetricsVersion compute_metrics_vetorized_version_chosen;
	
	static cl_context ctx;
	static cl_command_queue queue;
	static size_t preferred_wg_size;

	
	static void Init(ComputeMetricsVersion compute_metrics_version = ComputeMetricsVersion::Latest);
	static void InitVectorized(VectorizedComputeMetricsVersion compute_metrics_version = VectorizedComputeMetricsVersion::Latest);
	static void Reset();
	static void Release();

	static cl_kernel GetEntryDiscoverKernel();
	static cl_kernel GetComputeMetricsKernel();
	static cl_kernel GetSortKernel(bool smallKernel = false);

	static void ReleaseEntryDiscoverKernel();
	static void ReleaseComputeMetricsKernel();
	static void ReleaseSortKernel();
};