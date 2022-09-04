#pragma once
#include "dag.h"
#include "ocl_boiler.h"
#include "app_globals.h"
#include <tuple>

class processor_assignment
{
public:
	static cl_event* ScheduleTasksOnProcessors(Graph<edge_t>* DAG, metrics_t* ordered_metrics);
	static cl_event run_kernel(Graph<edge_t>* DAG, int current_node, cl_int3* task_processor_assignment, int* processorsNextSlotStart,
		int predecessor_with_max_aft, int weight_for_max_aft_predecessor, int max_aft_of_predecessors, edge_t* costs, 
		int processor_for_max_aft_predecessor, int cost_of_predecessors_in_different_processors);
};

