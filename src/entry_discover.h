#pragma once
#include "dag.h"
#include "ocl_boiler.h"
#include <tuple>


class EntryDiscover {
private:
	static cl_event run_entry_discover_kernel(int n_nodes);
	static tuple<cl_event, int*> entry_discover(Graph<int>* DAG);
public:
	static tuple<cl_event, int*> Run(Graph<int>* DAG);
};