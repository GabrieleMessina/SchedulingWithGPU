#pragma once
#include "ocl_boiler.h"
#include "ocl_manager.h"
#include <cmath>
#include "utils.h"

class OCLBufferManager {

public:
	static OCLBufferManager *instance;
	static OCLManager CLManager;
	cl_int err;

	int n_nodes;

	cl_mem graph_edges;
	size_t edges_memsize;
	cl_event read_edges_evt;
	cl_event write_edges_evt;

	cl_mem graph_edges_weights_reverse;
	size_t edges_weights_reverse_memsize;
	cl_event read_edges_weights_reverse_evt;
	cl_event write_edges_weights_reverse_evt;

	cl_mem graph_reverse_edges;
	size_t edges_reverse_memsize;
	cl_event read_edges_reverse_evt;
	cl_event write_edges_reverse_evt;
	
	cl_mem n_entrypoints;
	size_t n_entrypoints_memsize;
	cl_event read_nentries_evt;
	cl_event write_nentries_evt;
	
	cl_mem entrypoints;
	size_t entrypoints_memsize;
	cl_event read_entries_evt;
	cl_event write_entries_evt;
	
	cl_mem queue;
	size_t queue_memsize;
	cl_event read_queue_evt;
	cl_event write_queue_evt;	

	cl_mem next_queue;
	size_t next_queue_memsize;
	cl_event read_next_queue_evt;
	cl_event write_next_queue_evt;
	
	cl_mem metrics;
	size_t metrics_memsize;
	cl_event read_metrics_evt;
	cl_event write_metrics_evt;

	cl_mem ordered_metrics;
	size_t ordered_metrics_memsize;
	cl_event read_ordered_metrics_evt;
	cl_event write_ordered_metrics_evt;
	
	cl_mem nodes;
	size_t nodes_memsize;
	cl_event read_nodes_evt;
	cl_event write_nodes_evt;
	
	/*cl_mem local_queue;
	size_t local_queue_memsize;
	cl_event write_local_queue_evt;

	cl_mem local_queue_temp;
	size_t local_queue_temp_memsize;
	cl_event write_local_queue_temp_evt;*/

	cl_mem processors_cost;
	size_t processors_cost_memsize;
	cl_event read_processors_cost_evt;
	cl_event write_processors_cost_evt;
	
	cl_mem task_processor_assignment;
	size_t task_processor_assignment_memsize;
	cl_event read_task_processor_assignment_evt;
	cl_event write_task_processor_assignment_evt;
	
	cl_mem processors_next_slot_start;
	size_t processors_next_slot_start_memsize;
	cl_event read_processors_next_slot_start_evt;
	cl_event write_processors_next_slot_start_evt;
	
	cl_mem costs_on_processor;
	size_t costs_on_processor_memsize;
	cl_event read_costs_on_processor_evt;
	cl_event write_costs_on_processor_evt;

	static OCLBufferManager* Init(int nNodes, int adjSize, int adjReverseSize, int number_of_processors, bool vectorized = false);
	void Release();
	~OCLBufferManager();

	static OCLBufferManager *GetInstance();

	void SwapQueues();
	void SwapMetrics();

	void InitGraphEdges();
	void InitGraphWeightsReverse();
	void InitGraphReverseEdges();
	void InitNEntrypoints();
	void InitEntrypoints();
	void InitQueue();
	void InitNextQueue();
	void InitOrderedMetrics();
	void InitMetrics();
	void InitNodes();
	void InitProcessorsCost();
	void InitTaskProcessorAssignment();
	void InitProcessorNextSlotStart();
	void InitCostsOnProcessor();
	/*void InitLocalQueue();
	void InitLocalQueueTemp();*/

	cl_mem GetGraphEdges();
	cl_mem GetGraphWeightsReverse();
	cl_mem GetGraphReverseEdges();
	cl_mem GetNEntrypoints();
	cl_mem GetEntrypoints();
	cl_mem GetQueue();
	cl_mem GetNextQueue();
	cl_mem GetOrderedMetrics();
	cl_mem GetMetrics();
	cl_mem GetNodes();
	cl_mem GetProcessorsCost();
	cl_mem GetTaskProcessorAssignment();
	cl_mem GetProcessorNextSlotStart();
	cl_mem GetCostsOnProcessor();
	/*cl_mem GetLocalQueue();
	cl_mem GetLocalQueueTemp();*/

	void SetGraphEdges(const void* adj);
	void SetGraphWeightsReverse(const void* weights);
	void SetGraphReverseEdges(const void* adj);
	void SetNEntrypoints(const void* nEntrypoints);
	void SetEntrypoints(const void* entrypoints);
	void SetQueue(const int* queue);
	void SetQueue(const cl_int8* queue);
	void SetQueue(const cl_int4* queue);
	void SetNextQueue(const void* nextQueue);
	void SetQueue(const cl_int4* queue, void* out);
	void SetNextQueue(const void* nextQueue, void* out);
	void SetMetrics(const void* metrics);
	void SetOrderedMetrics(const void* metrics);
	void SetNodes(const void* nodes);
	void SetProcessorsCost(const void* data);
	void SetTaskProcessorAssignment(const void* data);
	void SetProcessorNextSlotStart(const void* data);
	void SetCostsOnProcessor(const void* data);
	/*void SetLocalQueue(const void* queue);
	void SetLocalQueueTemp(const void* queue);*/
	/*void SetGraphEdges(bool *adj);
	void SetNEntrypoints(int *nEntrypoints);
	void SetEntrypoints(int *entrypoints);
	void SetQueue(int* queue);
	void SetNextQueue(int* nextQueue);
	void SetMetrics(metrics_t* metrics);
	void SetOrderedMetrics(metrics_t* metrics);
	void SetNodes(int *nodes);*/

	void GetGraphEdgesResult(void* out, cl_event* eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetGraphWeightsResult(void* out, cl_event* eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetGraphEdgesReverseResult(void* out, cl_event* eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetNEntrypointsResult(void* out, cl_event *eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetEntrypointsResult(void* out, cl_event *eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetQueueResult(void* out, cl_event* eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetNextQueueResult(void* out, cl_event* eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetMetricsResult(void* out, cl_event* eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetOrderedMetricsResult(void* out, cl_event* eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetNodesResult(void* out, cl_event *eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetProcessorsCostResult(void* out, cl_event *eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetTaskProcessorAssignmentResult(void* out, cl_event *eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetProcessorNextSlotStartResult(void* out, cl_event *eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetCostsOnProcessorResult(void* out, cl_event *eventToWait = NULL, int numberOfEventsToWait = 0);
	/*void GetGraphEdgesResult(bool *out, cl_event* eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetNEntrypointsResult(int* out, cl_event *eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetEntrypointsResult(int* out, cl_event *eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetQueueResult(int* out, cl_event* eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetNextQueueResult(int* out, cl_event* eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetMetricsResult(metrics_t* out, cl_event* eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetOrderedMetricsResult(metrics_t* out, cl_event* eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetNodesResult(int* out, cl_event *eventToWait = NULL, int numberOfEventsToWait = 0);*/

	void ReleaseGraphEdges();
	void ReleaseGraphWeightsReverse();
	void ReleaseGraphReverseEdges();
	void ReleaseNEntrypoints();
	void ReleaseEntrypoints();
	void ReleaseQueue();
	void ReleaseNextQueue();
	void ReleaseQueue(void* data);
	void ReleaseNextQueue(void* data);
	void ReleaseMetrics();
	void ReleaseOrderedMetrics();
	void ReleaseNodes();
	void ReleaseProcessorsCost();
	void ReleaseTaskProcessorAssignment();
	void ReleaseProcessorNextSlotStart();
	void ReleaseCostsOnProcessor();
	/*void ReleaseLocalQueue();
	void ReleaseLocalQueueTemp();*/
};