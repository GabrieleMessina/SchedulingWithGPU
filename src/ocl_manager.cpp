#include "app_globals.h"
#include "ocl_manager.h"
#include "ocl_boiler.h"
#include "CL/cl.h"
#include "utils.h"


cl_int OCLManager::err;
cl_program OCLManager::prog;
cl_program OCLManager::prog2;

cl_kernel	OCLManager::entry_discover_k,
			OCLManager::compute_metrics_k,
			OCLManager::m_MergesortGlobalBigKernel,
			OCLManager::m_MergesortGlobalSmallKernel,
			OCLManager::m_MergesortStartKernel;

cl_context OCLManager::ctx;
cl_command_queue OCLManager::queue;
size_t OCLManager::preferred_wg_size;

void OCLManager::Init(const char* progName, const char* kernelNameEntryDiscover, const char* kernelNameComputeMetrics) {
	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	ctx = create_context(p, d);
	queue = create_queue(ctx, d);
	prog = create_program(progName, ctx, d);
	prog2 = create_program("./kernels/sort.ocl", ctx, d);

	entry_discover_k = clCreateKernel(prog, kernelNameEntryDiscover, &err);
	ocl_check(err, "create kernel %s", kernelNameEntryDiscover);
	compute_metrics_k = clCreateKernel(prog, kernelNameComputeMetrics, &err);
	ocl_check(err, "create kernel %s", kernelNameComputeMetrics);

	preferred_wg_size = get_preferred_work_group_size_multiple(compute_metrics_k, queue);

	m_MergesortStartKernel = clCreateKernel(prog2, "Sort_MergesortStart", &err);
	ocl_check(err, "create kernel %s", "Sort_MergesortStart");
	m_MergesortGlobalSmallKernel = clCreateKernel(prog2, "Sort_MergesortGlobalSmall", &err);
	ocl_check(err, "create kernel %s", "Sort_MergesortGlobalSmall");
	m_MergesortGlobalBigKernel = clCreateKernel(prog2, "Sort_MergesortGlobalBig", &err);
	ocl_check(err, "create kernel %s", "Sort_MergesortGlobalBig");
	//return *instance;
}

void OCLManager::Release() {
	clFinish(queue);
	clReleaseProgram(prog);
	clReleaseProgram(prog2);
	clReleaseContext(ctx);
	ReleaseEntryDiscoverKernel();
	ReleaseComputeMetricsKernel();
	ReleaseSortKernel();
}

void OCLManager::Reset() {
	clFinish(queue);
}


cl_kernel OCLManager::GetEntryDiscoverKernel() {
	return entry_discover_k;
}
cl_kernel OCLManager::GetComputeMetricsKernel() {
	return compute_metrics_k;
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
void OCLManager::ReleaseSortKernel() {
	clReleaseKernel(m_MergesortStartKernel);
	clReleaseKernel(m_MergesortGlobalSmallKernel);
	clReleaseKernel(m_MergesortGlobalBigKernel);
}
