#include "ocl_boiler.h"
#include "CL/cl.h"
#include "ocl_manager.h"
#include "ocl_buffer_manager.h"

OCLBufferManager *OCLBufferManager::instance = NULL;

OCLBufferManager OCLBufferManager::Init(int nNodes, bool vectorized) {
	OCLBufferManager* instance = new OCLBufferManager();
	instance->CLManager = *OCLManager::GetInstance();
	instance->n_nodes = nNodes;
	instance->edges_memsize = nNodes * nNodes * sizeof(bool);
	instance->entrypoints_memsize = nNodes * sizeof(int);
	instance->n_entrypoints_memsize = sizeof(int*);
	instance->nodes_memsize = nNodes * sizeof(int);
	instance->queue_memsize = (vectorized) ? nNodes * sizeof(cl_int4) : nNodes * sizeof(int);
	instance->next_queue_memsize = instance->queue_memsize;
	instance->edges_memsize = nNodes * nNodes * sizeof(bool);
	const int metrics_len = GetMetricsArrayLenght(nNodes); //necessario usare il round alla prossima potenza del due perché altrimenti il sort non potrebbe funzionare
	instance->metrics_memsize = metrics_len * sizeof(cl_int2);
	instance->ordered_metrics_memsize = instance->metrics_memsize;

	instance->InitGraphEdges();
	instance->InitNEntrypoints();
	instance->InitEntrypoints();
	instance->InitQueue();
	instance->InitNextQueue();
	instance->InitMetrics();
	instance->InitOrderedMetrics();
	instance->InitNodes();

	OCLBufferManager::instance = instance;
	return *instance;
}

OCLBufferManager *OCLBufferManager::GetInstance() {
	if (instance != NULL) return instance;
	error("Attemp to access uninitialized OCLBufferManager");
}

void OCLBufferManager::SwapQueues() {
	std::swap(queue, next_queue);
}
void OCLBufferManager::SwapMetrics() {
	std::swap(metrics, ordered_metrics);
}

/*Init*/
void OCLBufferManager::InitGraphEdges() {
	graph_edges = clCreateBuffer(CLManager.ctx, CL_MEM_READ_WRITE, edges_memsize, NULL, &err);
	ocl_check(err, "create buffer graph_edges");
}

void OCLBufferManager::InitNEntrypoints() {
	n_entrypoints = clCreateBuffer(CLManager.ctx, CL_MEM_WRITE_ONLY, n_entrypoints_memsize, NULL, &err);
	ocl_check(err, "create buffer n_entrypoints");
}
void OCLBufferManager::InitEntrypoints() {
	entrypoints = clCreateBuffer(CLManager.ctx, CL_MEM_READ_WRITE, entrypoints_memsize, NULL, &err);
	ocl_check(err, "create buffer entrypoints");
}
void OCLBufferManager::InitQueue() {
	queue = clCreateBuffer(CLManager.ctx, CL_MEM_READ_WRITE, queue_memsize, NULL, &err);
	ocl_check(err, "create buffer queue");
}
void OCLBufferManager::InitNextQueue() {
	next_queue = clCreateBuffer(CLManager.ctx, CL_MEM_READ_WRITE, next_queue_memsize, NULL, &err);
	ocl_check(err, "create buffer next_queue");
}
void OCLBufferManager::InitMetrics() {
	metrics = clCreateBuffer(CLManager.ctx, CL_MEM_READ_WRITE, metrics_memsize, NULL, &err);
	ocl_check(err, "create buffer metrics");
}
void OCLBufferManager::InitOrderedMetrics() {
	ordered_metrics = clCreateBuffer(CLManager.ctx, CL_MEM_READ_WRITE, ordered_metrics_memsize, NULL, &err);
	ocl_check(err, "create buffer Ordered metrics");
}
void OCLBufferManager::InitNodes() {
	nodes = clCreateBuffer(CLManager.ctx, CL_MEM_READ_WRITE, nodes_memsize, NULL, &err);
	ocl_check(err, "create buffer nodes");
}

/*Getter*/
cl_mem OCLBufferManager::GetGraphEdges() {
	return graph_edges;
}
cl_mem OCLBufferManager::GetNEntrypoints() {
	return n_entrypoints;
}
cl_mem OCLBufferManager::GetEntrypoints() {
	return entrypoints;
}
cl_mem OCLBufferManager::GetQueue() {
	return queue;
}
cl_mem OCLBufferManager::GetNextQueue() {
	return next_queue;
}
cl_mem OCLBufferManager::GetMetrics() {
	return metrics;
}
cl_mem OCLBufferManager::GetOrderedMetrics() {
	return ordered_metrics;
}
cl_mem OCLBufferManager::GetNodes() {
	return nodes;
}



/*Setter*/
void OCLBufferManager::SetGraphEdges(const void* adj) {
	err = clEnqueueWriteBuffer(CLManager.queue, GetGraphEdges(), CL_TRUE,
		0, edges_memsize, adj,
		0, NULL, &write_edges_evt);
	ocl_check(err, "write dataset edges into graph_edges");
}
void OCLBufferManager::SetNEntrypoints(const void* nEntries) {
	err = clEnqueueWriteBuffer(CLManager.queue, GetNEntrypoints(), CL_TRUE,
		0, n_entrypoints_memsize, nEntries,
		0, NULL, &write_nentries_evt);
	ocl_check(err, "write n_entries into n_entrypoints");
}
void OCLBufferManager::SetEntrypoints(const void* entries) {
	err = clEnqueueWriteBuffer(CLManager.queue, GetEntrypoints(), CL_TRUE,
		0, entrypoints_memsize, entries,
		0, NULL, &write_entries_evt);
	ocl_check(err, "write entries into entrypoints");
}
void OCLBufferManager::SetQueue(const void* queue) {
	err = clEnqueueWriteBuffer(CLManager.queue, GetQueue(), CL_TRUE,
		0, queue_memsize, queue,
		0, NULL, &write_queue_evt);
	ocl_check(err, "write into queue");
}
void OCLBufferManager::SetNextQueue(const void* nextQueue) {
	err = clEnqueueWriteBuffer(CLManager.queue, GetNextQueue(), CL_TRUE,
		0, next_queue_memsize, nextQueue,
		0, NULL, &write_next_queue_evt);
	ocl_check(err, "write into nextQueue");
}
void OCLBufferManager::SetMetrics(const void* metrics) {
	err = clEnqueueWriteBuffer(CLManager.queue, GetMetrics(), CL_TRUE,
		0, metrics_memsize, metrics,
		0, NULL, &write_metrics_evt);
	ocl_check(err, "write into metrics");
}
void OCLBufferManager::SetOrderedMetrics(const void* metrics) {
	err = clEnqueueWriteBuffer(CLManager.queue, GetOrderedMetrics(), CL_TRUE,
		0, ordered_metrics_memsize, metrics,
		0, NULL, &write_ordered_metrics_evt);
	ocl_check(err, "write into Ordered metrics");
}
void OCLBufferManager::SetNodes(const void* nodes) {
	err = clEnqueueWriteBuffer(CLManager.queue, GetNodes(), CL_TRUE,
		0, nodes_memsize, nodes,
		0, NULL, &write_nodes_evt);
	ocl_check(err, "write into nodes");
}


/*Result*/
void OCLBufferManager::GetGraphEdgesResult(void* out, cl_event *eventsToWait, int numberOfEventsToWait) {
	err = clEnqueueReadBuffer(CLManager.queue, GetGraphEdges(), CL_TRUE,
		0, edges_memsize, out,
		numberOfEventsToWait, eventsToWait, &read_edges_evt);
	ocl_check(err, "read buffer graph_edges");
}
void OCLBufferManager::GetNEntrypointsResult(void* out, cl_event *eventsToWait, int numberOfEventsToWait) {
	err = clEnqueueReadBuffer(CLManager.queue, GetNEntrypoints(), CL_TRUE,
		0, n_entrypoints_memsize, out,
		numberOfEventsToWait, eventsToWait, &read_nentries_evt);
	ocl_check(err, "read buffer n_entrypoints");
}
void OCLBufferManager::GetEntrypointsResult(void* out, cl_event *eventsToWait, int numberOfEventsToWait) {
	err = clEnqueueReadBuffer(CLManager.queue, GetEntrypoints(), CL_TRUE,
		0, entrypoints_memsize, out,
		numberOfEventsToWait, eventsToWait, &read_entries_evt);
	ocl_check(err, "read buffer entrypoints");
}
void OCLBufferManager::GetQueueResult(void* out, cl_event *eventsToWait, int numberOfEventsToWait) {
	err = clEnqueueReadBuffer(CLManager.queue, GetQueue(), CL_TRUE,
		0, queue_memsize, out,
		numberOfEventsToWait, eventsToWait, &read_queue_evt);
	ocl_check(err, "read buffer queue");
}
void OCLBufferManager::GetNextQueueResult(void* out, cl_event* eventsToWait, int numberOfEventsToWait) {
	err = clEnqueueReadBuffer(CLManager.queue, GetNextQueue(), CL_TRUE,
		0, next_queue_memsize, out,
		numberOfEventsToWait, eventsToWait, &read_next_queue_evt);
	ocl_check(err, "read buffer next_queue");
}
void OCLBufferManager::GetMetricsResult(void* out, cl_event* eventsToWait, int numberOfEventsToWait) {
	err = clEnqueueReadBuffer(CLManager.queue, GetMetrics(), CL_TRUE,
		0, metrics_memsize, out,
		numberOfEventsToWait, eventsToWait, &read_metrics_evt);
	ocl_check(err, "read buffer metrics");
}
void OCLBufferManager::GetOrderedMetricsResult(void* out, cl_event* eventsToWait, int numberOfEventsToWait) {
	err = clEnqueueReadBuffer(CLManager.queue, GetOrderedMetrics(), CL_TRUE,
		0, ordered_metrics_memsize, out,
		numberOfEventsToWait, eventsToWait, &read_ordered_metrics_evt);
	ocl_check(err, "read buffer Ordered metrics");
}
void OCLBufferManager::GetNodesResult(void* out, cl_event* eventsToWait, int numberOfEventsToWait) {
	err = clEnqueueReadBuffer(CLManager.queue, GetNodes(), CL_TRUE,
		0, nodes_memsize, out,
		numberOfEventsToWait, eventsToWait, &read_nodes_evt);
	ocl_check(err, "read buffer nodes");
}

/*Release*/
void OCLBufferManager::ReleaseGraphEdges() {
	clReleaseMemObject(GetGraphEdges());
}
void OCLBufferManager::ReleaseNEntrypoints() {
	clReleaseMemObject(GetNEntrypoints());
}
void OCLBufferManager::ReleaseEntrypoints() {
	clReleaseMemObject(GetEntrypoints());
}
void OCLBufferManager::ReleaseQueue() {
	clReleaseMemObject(GetQueue());
}
void OCLBufferManager::ReleaseNextQueue() {
	clReleaseMemObject(GetNextQueue());
}
void OCLBufferManager::ReleaseMetrics() {
	clReleaseMemObject(GetMetrics());
}
void OCLBufferManager::ReleaseOrderedMetrics() {
	clReleaseMemObject(GetOrderedMetrics());
}
void OCLBufferManager::ReleaseNodes() {
	clReleaseMemObject(GetNodes());
}