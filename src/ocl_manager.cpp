#include "ocl_manager.h"
#include "ocl_boiler.h"
#include "utils.h"

OCLManager *OCLManager::instance = NULL;

OCLManager OCLManager::Init(const char* progName, const char* kernelNameEntryDiscover, const char* kernelNameComputeMetrics) {
	OCLManager *instance = new OCLManager();
	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	instance->ctx = create_context(p, d);
	instance->queue = create_queue(instance->ctx, d);
	cl_program prog = create_program(progName, instance->ctx, d);
	cl_program prog2 = create_program("./kernels/sort.ocl", instance->ctx, d);

	instance->entry_discover_k = clCreateKernel(prog, kernelNameEntryDiscover, &instance->err);
	ocl_check(instance->err, "create kernel %s", kernelNameEntryDiscover);
	instance->compute_metrics_k = clCreateKernel(prog, kernelNameComputeMetrics, &instance->err);
	ocl_check(instance->err, "create kernel %s", kernelNameComputeMetrics);

	instance->preferred_wg_size = get_preferred_work_group_size_multiple(instance->compute_metrics_k, instance->queue);

	instance->m_MergesortStartKernel = clCreateKernel(prog2, "Sort_MergesortStart", &instance->err);
	ocl_check(instance->err, "create kernel %s", "Sort_MergesortStart");
	instance->m_MergesortGlobalSmallKernel = clCreateKernel(prog2, "Sort_MergesortGlobalSmall", &instance->err);
	ocl_check(instance->err, "create kernel %s", "Sort_MergesortGlobalSmall");
	instance->m_MergesortGlobalBigKernel = clCreateKernel(prog2, "Sort_MergesortGlobalBig", &instance->err);
	ocl_check(instance->err, "create kernel %s", "Sort_MergesortGlobalBig");

	OCLManager::instance = instance;
	return *instance;
}

OCLManager::~OCLManager() {
	/*ReleaseEntryDiscoverKernel();
	ReleaseComputeMetricsKernel();
	ReleaseSortKernel();*/
}

void OCLManager::Release() {
	delete instance;
}

OCLManager *OCLManager::GetInstance() {
	if (instance != NULL) return instance;
	error("Attemp to access uninitialized OCLManager");
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
