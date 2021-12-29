#pragma once
#include "utils.h"
#include "ocl_boiler.h"
#include "ocl_manager.h"

class OCLBufferManager {

private:
	static OCLBufferManager *instance;
	OCLManager CLManager;
	cl_int err;

	int n_nodes;

	cl_mem graph_edges;
	size_t edges_memsize;
	cl_event read_edges_evt;
	cl_event write_edges_evt;
	
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
	
public:
	static OCLBufferManager Init(int nNodes, bool vectorized = false);

	static OCLBufferManager *GetInstance();

	void SwapQueues();
	void SwapMetrics();

	void InitGraphEdges();
	void InitNEntrypoints();
	void InitEntrypoints();
	void InitQueue();
	void InitNextQueue();
	void InitOrderedMetrics();
	void InitMetrics();
	void InitNodes();

	cl_mem GetGraphEdges();
	cl_mem GetNEntrypoints();
	cl_mem GetEntrypoints();
	cl_mem GetQueue();
	cl_mem GetNextQueue();
	cl_mem GetOrderedMetrics();
	cl_mem GetMetrics();
	cl_mem GetNodes();

	void SetGraphEdges(const void* adj);
	void SetNEntrypoints(const void* nEntrypoints);
	void SetEntrypoints(const void* entrypoints);
	void SetQueue(const void* queue);
	void SetNextQueue(const void* nextQueue);
	void SetMetrics(const void* metrics);
	void SetOrderedMetrics(const void* metrics);
	void SetNodes(const void* nodes);
	/*void SetGraphEdges(bool *adj);
	void SetNEntrypoints(int *nEntrypoints);
	void SetEntrypoints(int *entrypoints);
	void SetQueue(int* queue);
	void SetNextQueue(int* nextQueue);
	void SetMetrics(cl_int2* metrics);
	void SetOrderedMetrics(cl_int2* metrics);
	void SetNodes(int *nodes);*/

	void GetGraphEdgesResult(void* out, cl_event* eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetNEntrypointsResult(void* out, cl_event *eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetEntrypointsResult(void* out, cl_event *eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetQueueResult(void* out, cl_event* eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetNextQueueResult(void* out, cl_event* eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetMetricsResult(void* out, cl_event* eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetOrderedMetricsResult(void* out, cl_event* eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetNodesResult(void* out, cl_event *eventToWait = NULL, int numberOfEventsToWait = 0);
	/*void GetGraphEdgesResult(bool *out, cl_event* eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetNEntrypointsResult(int* out, cl_event *eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetEntrypointsResult(int* out, cl_event *eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetQueueResult(int* out, cl_event* eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetNextQueueResult(int* out, cl_event* eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetMetricsResult(cl_int2* out, cl_event* eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetOrderedMetricsResult(cl_int2* out, cl_event* eventToWait = NULL, int numberOfEventsToWait = 0);
	void GetNodesResult(int* out, cl_event *eventToWait = NULL, int numberOfEventsToWait = 0);*/

	void ReleaseGraphEdges();
	void ReleaseNEntrypoints();
	void ReleaseEntrypoints();
	void ReleaseQueue();
	void ReleaseNextQueue();
	void ReleaseMetrics();
	void ReleaseOrderedMetrics();
	void ReleaseNodes();
};