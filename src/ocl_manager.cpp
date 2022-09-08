#include "app_globals.h"
#include "ocl_manager.h"
#include "ocl_boiler.h"
#include "CL/cl.h"
#include "utils.h"


cl_int OCLManager::err;
cl_program OCLManager::entry_discover_prog;
cl_program OCLManager::compute_metrics_prog;
cl_program OCLManager::reduce_queue_prog;
cl_program OCLManager::sort_prog;

cl_kernel	OCLManager::entry_discover_k,
			OCLManager::compute_metrics_k,
			OCLManager::reset_k,
			OCLManager::reduce_queue_k,
			OCLManager::compute_processor_cost_k,
			OCLManager::m_MergesortGlobalBigKernel,
			OCLManager::m_MergesortGlobalSmallKernel,
			OCLManager::m_MergesortStartKernel;

cl_context OCLManager::ctx;
cl_command_queue OCLManager::queue;
size_t OCLManager::preferred_wg_size;

ComputeMetricsVersion OCLManager::compute_metrics_version_chosen;
VectorizedComputeMetricsVersion OCLManager::compute_metrics_vetorized_version_chosen;

void OCLManager::InitCommon(const char* entryDiscoverKernelName, const char* computeMetricsKernelName, const char* reductionKernelName, const char* sortKernelName) {
	cl_platform_id p = select_platform(); //49MB
	cl_device_id d = select_device(p); //0MB
	ctx = create_context(p, d); //103MB
	queue = create_queue(ctx, d); //0MB
	entry_discover_prog = create_program("./kernels/entry_discover.ocl", ctx, d); //165MB
	compute_metrics_prog = create_program("./kernels/compute_metrics.ocl", ctx, d);
	reduce_queue_prog = create_program("./kernels/reduction.ocl", ctx, d);
	sort_prog = create_program("./kernels/sort.ocl", ctx, d);

	entry_discover_k = clCreateKernel(entry_discover_prog, entryDiscoverKernelName, &err);
	ocl_check(err, "create kernel %s", entryDiscoverKernelName);
	compute_metrics_k = clCreateKernel(compute_metrics_prog, computeMetricsKernelName, &err);
	ocl_check(err, "create kernel %s", computeMetricsKernelName);
	reset_k = clCreateKernel(compute_metrics_prog, "reset", &err);
	ocl_check(err, "create kernel %s", "reset");
	reduce_queue_k = clCreateKernel(reduce_queue_prog, reductionKernelName, &err);
	ocl_check(err, "create kernel %s", reductionKernelName);
	compute_processor_cost_k = clCreateKernel(reduce_queue_prog, "compute_processor_cost", &err);
	ocl_check(err, "create kernel %s", "compute_processor_cost");

	preferred_wg_size = get_preferred_work_group_size_multiple(compute_metrics_k, queue);

	m_MergesortStartKernel = clCreateKernel(sort_prog, "Sort_MergesortStart", &err);
	ocl_check(err, "create kernel %s", "Sort_MergesortStart");
	m_MergesortGlobalSmallKernel = clCreateKernel(sort_prog, "Sort_MergesortGlobalSmall", &err);
	ocl_check(err, "create kernel %s", "Sort_MergesortGlobalSmall");
	m_MergesortGlobalBigKernel = clCreateKernel(sort_prog, "Sort_MergesortGlobalBig", &err);
	ocl_check(err, "create kernel %s", "Sort_MergesortGlobalBig");
}

void OCLManager::Init(ComputeMetricsVersion compute_metrics_version) {
	compute_metrics_version_chosen = compute_metrics_version;
	switch (compute_metrics_version_chosen)
	{
	case ComputeMetricsVersion::Latest:
		compute_metrics_version_chosen = ComputeMetricsVersion::Rectangular;
	case ComputeMetricsVersion::Working:
		compute_metrics_version_chosen = ComputeMetricsVersion::Rectangular;
	case ComputeMetricsVersion::Rectangular:
		InitCommon("entry_discover", "compute_metrics_standard_with_rectangular_matrix", "reduce_queue", "sort_kernel");
		break;
	case ComputeMetricsVersion::v1:
	default:
		InitCommon("entry_discover", "compute_metrics_second", "reduce_queue", "sort_kernel");
		break;
	}
}


void OCLManager::InitVectorized(VectorizedComputeMetricsVersion compute_metrics_version) {
	compute_metrics_vetorized_version_chosen = compute_metrics_version;
	switch (compute_metrics_vetorized_version_chosen)
	{
	case VectorizedComputeMetricsVersion::Latest:
		compute_metrics_vetorized_version_chosen = VectorizedComputeMetricsVersion::RectangularV2;
	case VectorizedComputeMetricsVersion::Working:
		compute_metrics_vetorized_version_chosen = VectorizedComputeMetricsVersion::RectangularV2;
	case VectorizedComputeMetricsVersion::RectangularVec8:
		InitCommon("entry_discover", "compute_metrics_vectorized8_rectangular", "reduce_queue", "sort_kernel");
		break;
	case VectorizedComputeMetricsVersion::RectangularV2:
		InitCommon("entry_discover", "compute_metrics_vectorized_rectangular_v2", "reduce_queue", "sort_kernel");
		break;
	case VectorizedComputeMetricsVersion::Rectangular:
		InitCommon("entry_discover", "compute_metrics_vectorized_rectangular", "reduce_queue", "sort_kernel");
		break;
	case VectorizedComputeMetricsVersion::v2:
		InitCommon("entry_discover", "compute_metrics_eighth", "reduce_queue", "sort_kernel");
		break;
	case VectorizedComputeMetricsVersion::v1:
	default:
		InitCommon("entry_discover", "compute_metrics_fifth", "reduce_queue", "sort_kernel");
		break;
	}
}

void OCLManager::Release() {
	clFinish(queue);
	ReleaseEntryDiscoverKernel();
	ReleaseComputeMetricsKernel();
	ReleaseReduceQueueKernel();
	ReleaseResetKernel();
	ReleaseSortKernel();
	ReleaseComputeProcessorCostKernel();
	clReleaseProgram(entry_discover_prog);
	clReleaseProgram(compute_metrics_prog);
	clReleaseProgram(reduce_queue_prog);
	clReleaseProgram(sort_prog);
	clReleaseCommandQueue(queue);
	clReleaseContext(ctx);
}

void OCLManager::Reset() {
	err = clFinish(queue);
	ocl_check(err, "clFinish error.");
}


cl_kernel OCLManager::GetEntryDiscoverKernel() {
	return entry_discover_k;
}
cl_kernel OCLManager::GetComputeMetricsKernel() {
	return compute_metrics_k;
}
cl_kernel OCLManager::GetResetKernel() {
	return reset_k;
}
cl_kernel OCLManager::GetReduceQueueKernel() {
	return reduce_queue_k;
}
cl_kernel OCLManager::GetComputeProcessorCostKernel() {
	return compute_processor_cost_k;
}
cl_kernel OCLManager::GetSortKernel(bool smallKernel) {
	if(smallKernel)
		return m_MergesortGlobalSmallKernel;
	else 
		return m_MergesortGlobalBigKernel;
	//clReleaseKernel(OCLManager::m_MergesortStartKernel);
}


void OCLManager::ReleaseEntryDiscoverKernel() {
	clReleaseKernel(entry_discover_k);
}
void OCLManager::ReleaseComputeMetricsKernel() {
	clReleaseKernel(compute_metrics_k);
}
void OCLManager::ReleaseResetKernel() {
	clReleaseKernel(reset_k);
}
void OCLManager::ReleaseReduceQueueKernel() {
	clReleaseKernel(reduce_queue_k);
}
void OCLManager::ReleaseComputeProcessorCostKernel() {
	clReleaseKernel(compute_processor_cost_k);
}
void OCLManager::ReleaseSortKernel() {
	clReleaseKernel(m_MergesortStartKernel);
	clReleaseKernel(m_MergesortGlobalSmallKernel);
	clReleaseKernel(m_MergesortGlobalBigKernel);
}
