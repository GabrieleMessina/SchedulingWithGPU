#pragma once
#include "dag.h"
#include "ocl_boiler.h"
#include "app_globals.h"
#include <tuple>

class processor_assignment
{
public:
	static void ScheduleTasksOnProcessors(Graph<edge_t>* DAG, metrics_t* ordered_metrics);
};

