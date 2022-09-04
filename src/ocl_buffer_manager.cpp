#include "app_globals.h"
#include "ocl_boiler.h"
#include "CL/cl.h"
#include "ocl_manager.h"
#include "ocl_buffer_manager.h"

OCLBufferManager *OCLBufferManager::instance = NULL;
OCLManager OCLBufferManager::CLManager;

OCLBufferManager OCLBufferManager::Init(int nNodes, int adjSize, int adjReverseSize, int number_of_processors, bool vectorized) {
	OCLBufferManager* instance = DBG_NEW OCLBufferManager();
	instance->n_nodes = nNodes;
	instance->edges_memsize = adjSize * sizeof(edge_t);
	instance->edges_reverse_memsize = adjReverseSize * sizeof(edge_t);
	instance->edges_weights_reverse_memsize = instance->edges_reverse_memsize;
	instance->entrypoints_memsize = nNodes * sizeof(int);
	instance->n_entrypoints_memsize = sizeof(int);
	instance->nodes_memsize = nNodes * sizeof(int);
	instance->queue_memsize = (vectorized) ? ceil(nNodes / 4.0) * sizeof(cl_int4) : nNodes * sizeof(int);
	if(OCLManager::compute_metrics_vetorized_version_chosen == VectorizedComputeMetricsVersion::RectangularVec8)
		instance->queue_memsize = (vectorized) ? ceil(nNodes / 8.0) * sizeof(cl_int8) : nNodes * sizeof(int);
	instance->next_queue_memsize = instance->queue_memsize;
	const int metrics_len = GetMetricsArrayLenght(nNodes); //necessario usare il round alla prossima potenza del due perché altrimenti il sort non potrebbe funzionare
	instance->metrics_memsize = metrics_len * sizeof(metrics_t);
	instance->ordered_metrics_memsize = instance->metrics_memsize;
	//instance->local_queue_memsize = (vectorized) ? CLManager.preferred_wg_size * sizeof(cl_int4) : nNodes * sizeof(int); //TODO: min(preferred e n_nodes/4);
	
	instance->processors_cost_memsize = number_of_processors * sizeof(cl_int3);
	instance->task_processor_assignment_memsize = nNodes * sizeof(cl_int3);
	instance->processors_next_slot_start_memsize = number_of_processors * sizeof(int);
	instance->costs_on_processor_memsize = nNodes * number_of_processors * sizeof(edge_t);

	
	
	instance->InitGraphEdges();
	instance->InitGraphWeightsReverse();
	instance->InitGraphReverseEdges();
	instance->InitNEntrypoints();
	instance->InitEntrypoints();
	instance->InitQueue();
	instance->InitNextQueue();
	instance->InitMetrics();
	instance->InitOrderedMetrics();
	instance->InitNodes();
	instance->InitProcessorsCost();
	instance->InitTaskProcessorAssignment();
	instance->InitProcessorNextSlotStart();
	instance->InitCostsOnProcessor();
	/*instance->InitLocalQueue();
	instance->InitLocalQueueTemp();*/

	OCLBufferManager::instance = instance;
	return *instance;
}

OCLBufferManager::~OCLBufferManager() {
	/*free(metrics);
	free(queue);
	free(next_queue);*/
	/*ReleaseGraphEdges();
	ReleaseNEntrypoints();
	ReleaseEntrypoints();
	ReleaseQueue();
	ReleaseNextQueue();
	ReleaseMetrics();
	ReleaseOrderedMetrics();
	ReleaseNodes();*/
}
void OCLBufferManager::Release () {
	if (instance != NULL) delete instance;
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
	graph_edges = clCreateBuffer(OCLManager::ctx, CL_MEM_READ_WRITE, edges_memsize, NULL, &err);
	ocl_check(err, "create buffer graph_edges");
}
void OCLBufferManager::InitGraphWeightsReverse() {
	graph_edges_weights_reverse = clCreateBuffer(OCLManager::ctx, CL_MEM_READ_WRITE, edges_weights_reverse_memsize, NULL, &err);
	ocl_check(err, "create buffer graph_edges_weights");
}
void OCLBufferManager::InitGraphReverseEdges() {
	graph_reverse_edges = clCreateBuffer(OCLManager::ctx, CL_MEM_READ_WRITE, edges_reverse_memsize, NULL, &err);
	ocl_check(err, "create buffer graph_reverse_edges");
}
void OCLBufferManager::InitNEntrypoints() {
	n_entrypoints = clCreateBuffer(OCLManager::ctx, CL_MEM_WRITE_ONLY, n_entrypoints_memsize, NULL, &err);
	ocl_check(err, "create buffer n_entrypoints");
}
void OCLBufferManager::InitEntrypoints() {
	entrypoints = clCreateBuffer(OCLManager::ctx, CL_MEM_READ_WRITE, entrypoints_memsize, NULL, &err);
	ocl_check(err, "create buffer entrypoints");
}
void OCLBufferManager::InitQueue() {
	queue = clCreateBuffer(OCLManager::ctx, CL_MEM_READ_WRITE, queue_memsize, NULL, &err);
	ocl_check(err, "create buffer queue");
}
void OCLBufferManager::InitNextQueue() {
	next_queue = clCreateBuffer(OCLManager::ctx, CL_MEM_READ_WRITE, next_queue_memsize, NULL, &err);
	ocl_check(err, "create buffer next_queue");
}
void OCLBufferManager::InitMetrics() {
	metrics = clCreateBuffer(OCLManager::ctx, CL_MEM_READ_WRITE, metrics_memsize, NULL, &err);
	ocl_check(err, "create buffer metrics");
}
void OCLBufferManager::InitOrderedMetrics() {
	ordered_metrics = clCreateBuffer(OCLManager::ctx, CL_MEM_READ_WRITE, ordered_metrics_memsize, NULL, &err);
	ocl_check(err, "create buffer Ordered metrics");
}
void OCLBufferManager::InitNodes() {
	nodes = clCreateBuffer(OCLManager::ctx, CL_MEM_READ_WRITE, nodes_memsize, NULL, &err);
	ocl_check(err, "create buffer nodes");
}
void OCLBufferManager::InitProcessorsCost() {
	processors_cost = clCreateBuffer(OCLManager::ctx, CL_MEM_READ_WRITE, processors_cost_memsize, NULL, &err);
	ocl_check(err, "create buffer processors_cost");
}
void OCLBufferManager::InitTaskProcessorAssignment() {
	task_processor_assignment = clCreateBuffer(OCLManager::ctx, CL_MEM_READ_WRITE, task_processor_assignment_memsize, NULL, &err);
	ocl_check(err, "create buffer task_processor_assignment");
}
void OCLBufferManager::InitProcessorNextSlotStart() {
	processors_next_slot_start = clCreateBuffer(OCLManager::ctx, CL_MEM_READ_WRITE, processors_next_slot_start_memsize, NULL, &err);
	ocl_check(err, "create buffer processors_next_slot_start");
}
void OCLBufferManager::InitCostsOnProcessor() {
	costs_on_processor = clCreateBuffer(OCLManager::ctx, CL_MEM_READ_WRITE, costs_on_processor_memsize, NULL, &err);
	ocl_check(err, "create buffer costs_on_processor");
}
//void OCLBufferManager::InitLocalQueue() {
//	local_queue = clCreateBuffer(OCLManager::ctx, CL_MEM_READ_WRITE, local_queue_memsize, NULL, &err);
//	ocl_check(err, "create local buffer queue");
//}
//void OCLBufferManager::InitLocalQueueTemp() {
//	local_queue_temp = clCreateBuffer(OCLManager::ctx, CL_MEM_READ_WRITE, local_queue_memsize, NULL, &err);
//	ocl_check(err, "create local buffer queue temp");
//}

/*Getter*/
cl_mem OCLBufferManager::GetGraphEdges() {
	return graph_edges;
}
cl_mem OCLBufferManager::GetGraphWeightsReverse() {
	return graph_edges_weights_reverse;
}
cl_mem OCLBufferManager::GetGraphReverseEdges() {
	return graph_reverse_edges;
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
cl_mem OCLBufferManager::GetProcessorsCost() {
	return processors_cost;
}
cl_mem OCLBufferManager::GetTaskProcessorAssignment() {
	return task_processor_assignment;
}
cl_mem OCLBufferManager::GetProcessorNextSlotStart() {
	return processors_next_slot_start;
}
cl_mem OCLBufferManager::GetCostsOnProcessor() {
	return costs_on_processor;
}
//cl_mem OCLBufferManager::GetLocalQueue() {
//	return local_queue;
//}
//cl_mem OCLBufferManager::GetLocalQueueTemp() {
//	return local_queue_temp;
//}



/*Setter*/
void OCLBufferManager::SetGraphEdges(const void* adj) {
	err = clEnqueueWriteBuffer(OCLManager::queue, GetGraphEdges(), CL_TRUE,
		0, edges_memsize, adj,
		0, NULL, &write_edges_evt);
	ocl_check(err, "write dataset edges into graph_edges");
}
void OCLBufferManager::SetGraphWeightsReverse(const void* weights) {
	err = clEnqueueWriteBuffer(OCLManager::queue, GetGraphWeightsReverse(), CL_TRUE,
		0, edges_weights_reverse_memsize, weights,
		0, NULL, &write_edges_weights_reverse_evt);
	ocl_check(err, "write dataset weights into graph_edges");
}
void OCLBufferManager::SetGraphReverseEdges(const void* adj) {
	err = clEnqueueWriteBuffer(OCLManager::queue, GetGraphReverseEdges(), CL_TRUE,
		0, edges_reverse_memsize, adj,
		0, NULL, &write_edges_reverse_evt);
	ocl_check(err, "write dataset edges reverse into graph_edges");
}
void OCLBufferManager::SetNEntrypoints(const void* nEntries) {
	err = clEnqueueWriteBuffer(OCLManager::queue, GetNEntrypoints(), CL_TRUE,
		0, n_entrypoints_memsize, nEntries,
		0, NULL, &write_nentries_evt);
	ocl_check(err, "write n_entries into n_entrypoints");
}
void OCLBufferManager::SetEntrypoints(const void* entries) {
	err = clEnqueueWriteBuffer(OCLManager::queue, GetEntrypoints(), CL_TRUE,
		0, entrypoints_memsize, entries,
		0, NULL, &write_entries_evt);
	ocl_check(err, "write entries into entrypoints");
}
void OCLBufferManager::SetQueue(const cl_int8* queue) {
	err = clEnqueueWriteBuffer(OCLManager::queue, GetQueue(), CL_TRUE,
		0, queue_memsize, queue,
		0, NULL, &write_queue_evt);
	ocl_check(err, "write into queue");
}
void OCLBufferManager::SetQueue(const cl_int4* queue) {
	err = clEnqueueWriteBuffer(OCLManager::queue, GetQueue(), CL_TRUE,
		0, queue_memsize, queue,
		0, NULL, &write_queue_evt);
	ocl_check(err, "write into queue");
}
void OCLBufferManager::SetQueue(const int* queue) {
	err = clEnqueueWriteBuffer(OCLManager::queue, GetQueue(), CL_TRUE,
		0, queue_memsize, queue,
		0, NULL, &write_queue_evt);
	ocl_check(err, "write into queue");
}
void OCLBufferManager::SetNextQueue(const void* nextQueue) {
	err = clEnqueueWriteBuffer(OCLManager::queue, GetNextQueue(), CL_TRUE,
		0, next_queue_memsize, nextQueue,
		0, NULL, &write_next_queue_evt);
	ocl_check(err, "write into nextQueue");
}
void OCLBufferManager::SetQueue(const cl_int4* queue, void* out) {//TODO: queue is useless here
	out = clEnqueueMapBuffer(OCLManager::queue, GetQueue(), CL_TRUE, CL_MAP_READ,
		0, queue_memsize,
		0, NULL, &write_queue_evt, &err);
	ocl_check(err, "write into queue with map buffer");
}
void OCLBufferManager::SetNextQueue(const void* nextQueue, void* out) {
	out = clEnqueueMapBuffer(OCLManager::queue, GetNextQueue(), CL_TRUE, CL_MAP_READ,
		0, next_queue_memsize,
		0, NULL, &write_next_queue_evt, &err);
	ocl_check(err, "write into nextQueue with map buffer");
}
void OCLBufferManager::SetMetrics(const void* metrics) {
	err = clEnqueueWriteBuffer(OCLManager::queue, GetMetrics(), CL_TRUE,
		0, metrics_memsize, metrics,
		0, NULL, &write_metrics_evt);
	ocl_check(err, "write into metrics");
}
void OCLBufferManager::SetOrderedMetrics(const void* metrics) {
	err = clEnqueueWriteBuffer(OCLManager::queue, GetOrderedMetrics(), CL_TRUE,
		0, ordered_metrics_memsize, metrics,
		0, NULL, &write_ordered_metrics_evt);
	ocl_check(err, "write into Ordered metrics");
}
void OCLBufferManager::SetNodes(const void* nodes) {
	err = clEnqueueWriteBuffer(OCLManager::queue, GetNodes(), CL_TRUE,
		0, nodes_memsize, nodes,
		0, NULL, &write_nodes_evt);
	ocl_check(err, "write into nodes");
}
void OCLBufferManager::SetProcessorsCost(const void* data) {
	err = clEnqueueWriteBuffer(OCLManager::queue, GetProcessorsCost(), CL_TRUE,
		0, processors_cost_memsize, data,
		0, NULL, &write_processors_cost_evt);
	ocl_check(err, "write into processors_cost");
}
void OCLBufferManager::SetTaskProcessorAssignment(const void* data) {
	err = clEnqueueWriteBuffer(OCLManager::queue, GetTaskProcessorAssignment(), CL_TRUE,
		0, task_processor_assignment_memsize, data,
		0, NULL, &write_task_processor_assignment_evt);
	ocl_check(err, "write into task_processor_assignment");
}
void OCLBufferManager::SetProcessorNextSlotStart(const void* data) {
	err = clEnqueueWriteBuffer(OCLManager::queue, GetProcessorNextSlotStart(), CL_TRUE,
		0, processors_next_slot_start_memsize, data,
		0, NULL, &write_processors_next_slot_start_evt);
	ocl_check(err, "write into processors_next_slot_start");
}
void OCLBufferManager::SetCostsOnProcessor(const void* data) {
	err = clEnqueueWriteBuffer(OCLManager::queue, GetCostsOnProcessor(), CL_TRUE,
		0, costs_on_processor_memsize, data,
		0, NULL, &write_costs_on_processor_evt);
	ocl_check(err, "write into costs_on_processor");
}
//void OCLBufferManager::SetLocalQueue(const void* local_queue) {
//	err = clEnqueueWriteBuffer(OCLManager::queue, GetLocalQueue(), CL_TRUE,
//		0, local_queue_memsize, local_queue,
//		0, NULL, &write_local_queue_evt);
//	ocl_check(err, "write into queue");
//}
//void OCLBufferManager::SetLocalQueueTemp(const void* local_queue) {
//	err = clEnqueueWriteBuffer(OCLManager::queue, GetLocalQueueTemp(), CL_TRUE,
//		0, local_queue_memsize, local_queue,
//		0, NULL, &write_local_queue_temp_evt);
//	ocl_check(err, "write into queue");
//}


/*Result*/
void OCLBufferManager::GetGraphEdgesResult(void* out, cl_event *eventsToWait, int numberOfEventsToWait) {
	err = clEnqueueReadBuffer(OCLManager::queue, GetGraphEdges(), CL_TRUE,
		0, edges_memsize, out,
		numberOfEventsToWait, eventsToWait, &read_edges_evt);
	ocl_check(err, "read buffer graph_edges");
}
void OCLBufferManager::GetGraphWeightsResult(void* out, cl_event *eventsToWait, int numberOfEventsToWait) {
	err = clEnqueueReadBuffer(OCLManager::queue, GetGraphWeightsReverse(), CL_TRUE,
		0, edges_weights_reverse_memsize, out,
		numberOfEventsToWait, eventsToWait, &read_edges_weights_reverse_evt);
	ocl_check(err, "read buffer graph_edges_weights");
}
void OCLBufferManager::GetGraphEdgesReverseResult(void* out, cl_event *eventsToWait, int numberOfEventsToWait) {
	err = clEnqueueReadBuffer(OCLManager::queue, GetGraphReverseEdges(), CL_TRUE,
		0, edges_reverse_memsize, out,
		numberOfEventsToWait, eventsToWait, &read_edges_reverse_evt);
	ocl_check(err, "read buffer graph_edges");
}
void OCLBufferManager::GetNEntrypointsResult(void* out, cl_event *eventsToWait, int numberOfEventsToWait) {
	err = clEnqueueReadBuffer(OCLManager::queue, GetNEntrypoints(), CL_TRUE,
		0, n_entrypoints_memsize, out,
		numberOfEventsToWait, eventsToWait, &read_nentries_evt);
	ocl_check(err, "read buffer n_entrypoints");
}
void OCLBufferManager::GetEntrypointsResult(void* out, cl_event *eventsToWait, int numberOfEventsToWait) {
	err = clEnqueueReadBuffer(OCLManager::queue, GetEntrypoints(), CL_TRUE,
		0, entrypoints_memsize, out,
		numberOfEventsToWait, eventsToWait, &read_entries_evt);
	ocl_check(err, "read buffer entrypoints");
}
void OCLBufferManager::GetQueueResult(void* out, cl_event *eventsToWait, int numberOfEventsToWait) {
	err = clEnqueueReadBuffer(OCLManager::queue, GetQueue(), CL_TRUE,
		0, queue_memsize, out,
		numberOfEventsToWait, eventsToWait, &read_queue_evt);
	ocl_check(err, "read buffer queue");
}
void OCLBufferManager::GetNextQueueResult(void* out, cl_event* eventsToWait, int numberOfEventsToWait) {
	err = clEnqueueReadBuffer(OCLManager::queue, GetNextQueue(), CL_TRUE,
		0, next_queue_memsize, out,
		numberOfEventsToWait, eventsToWait, &read_next_queue_evt);
	ocl_check(err, "read buffer next_queue");
}
void OCLBufferManager::GetMetricsResult(void* out, cl_event* eventsToWait, int numberOfEventsToWait) {
	err = clEnqueueReadBuffer(OCLManager::queue, GetMetrics(), CL_TRUE,
		0, metrics_memsize, out,
		numberOfEventsToWait, eventsToWait, &read_metrics_evt);
	ocl_check(err, "read buffer metrics");
}
void OCLBufferManager::GetOrderedMetricsResult(void* out, cl_event* eventsToWait, int numberOfEventsToWait) {
	err = clEnqueueReadBuffer(OCLManager::queue, GetOrderedMetrics(), CL_TRUE,
		0, ordered_metrics_memsize, out,
		numberOfEventsToWait, eventsToWait, &read_ordered_metrics_evt);
	ocl_check(err, "read buffer Ordered metrics");
}
void OCLBufferManager::GetNodesResult(void* out, cl_event* eventsToWait, int numberOfEventsToWait) {
	err = clEnqueueReadBuffer(OCLManager::queue, GetNodes(), CL_TRUE,
		0, nodes_memsize, out,
		numberOfEventsToWait, eventsToWait, &read_nodes_evt);
	ocl_check(err, "read buffer nodes");
}
void OCLBufferManager::GetProcessorsCostResult(void* out, cl_event* eventsToWait, int numberOfEventsToWait) {
	err = clEnqueueReadBuffer(OCLManager::queue, GetProcessorsCost(), CL_TRUE,
		0, processors_cost_memsize, out,
		numberOfEventsToWait, eventsToWait, &read_processors_cost_evt);
	ocl_check(err, "read buffer processors_cost");
}
void OCLBufferManager::GetTaskProcessorAssignmentResult(void* out, cl_event* eventsToWait, int numberOfEventsToWait) {
	err = clEnqueueReadBuffer(OCLManager::queue, GetTaskProcessorAssignment(), CL_TRUE,
		0, task_processor_assignment_memsize, out,
		numberOfEventsToWait, eventsToWait, &read_task_processor_assignment_evt);
	ocl_check(err, "read buffer task_processor_assignment");
}
void OCLBufferManager::GetProcessorNextSlotStartResult(void* out, cl_event* eventsToWait, int numberOfEventsToWait) {
	err = clEnqueueReadBuffer(OCLManager::queue, GetProcessorNextSlotStart(), CL_TRUE,
		0, processors_next_slot_start_memsize, out,
		numberOfEventsToWait, eventsToWait, &read_processors_next_slot_start_evt);
	ocl_check(err, "read buffer processors_next_slot_start");
}
void OCLBufferManager::GetCostsOnProcessorResult(void* out, cl_event* eventsToWait, int numberOfEventsToWait) {
	err = clEnqueueReadBuffer(OCLManager::queue, GetCostsOnProcessor(), CL_TRUE,
		0, costs_on_processor_memsize, out,
		numberOfEventsToWait, eventsToWait, &read_costs_on_processor_evt);
	ocl_check(err, "read buffer costs_on_processor");
}

/*Release*/
void OCLBufferManager::ReleaseGraphEdges() {
	clReleaseMemObject(GetGraphEdges());
}
void OCLBufferManager::ReleaseGraphWeightsReverse() {
	clReleaseMemObject(GetGraphWeightsReverse());
}
void OCLBufferManager::ReleaseGraphReverseEdges() {
	clReleaseMemObject(GetGraphReverseEdges());
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
void OCLBufferManager::ReleaseQueue(void* data) {
	clEnqueueUnmapMemObject(OCLManager::queue, GetQueue(), data, 0, NULL, NULL);
}
void OCLBufferManager::ReleaseNextQueue(void* data) {
	clEnqueueUnmapMemObject(OCLManager::queue, GetNextQueue(), data, 0, NULL, NULL);
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
void OCLBufferManager::ReleaseProcessorsCost() {
	clReleaseMemObject(GetProcessorsCost());
}
void OCLBufferManager::ReleaseTaskProcessorAssignment() {
	clReleaseMemObject(GetTaskProcessorAssignment());
}
void OCLBufferManager::ReleaseProcessorNextSlotStart() {
	clReleaseMemObject(GetProcessorNextSlotStart());
}
void OCLBufferManager::ReleaseCostsOnProcessor() {
	clReleaseMemObject(GetCostsOnProcessor());
}
//void OCLBufferManager::ReleaseLocalQueue() {
//	clReleaseMemObject(GetLocalQueue());
//}
//void OCLBufferManager::ReleaseLocalQueueTemp() {
//	clReleaseMemObject(GetLocalQueueTemp());
//}