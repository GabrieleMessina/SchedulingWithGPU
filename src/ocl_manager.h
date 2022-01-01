#pragma once
#include "CL/cl.h"

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
	enum class Version { Latest, v1, v2, v3};
	static enum class VectorizedVersion { Latest, v1, v2, v3};
	static cl_context ctx;
	static cl_command_queue queue;
	static size_t preferred_wg_size;

	
	static void Init(Version version = Version::Latest);
	static void InitVectorized(VectorizedVersion version = VectorizedVersion::Latest);
	static void Reset();
	static void Release();

	static cl_kernel GetEntryDiscoverKernel();
	static cl_kernel GetComputeMetricsKernel();
	static cl_kernel GetSortKernel(bool smallKernel = false);

	static void ReleaseEntryDiscoverKernel();
	static void ReleaseComputeMetricsKernel();
	static void ReleaseSortKernel();
};