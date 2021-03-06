#pragma once
#include "dag.h"
#include "ocl_boiler.h"
#include "ocl_manager.h"
#include <tuple>


class ComputeMetrics {
private:
	static cl_event run_compute_metrics_kernel(int n_nodes, bool flip, Graph<edge_t>* DAG = NULL);
	static cl_event run_compute_metrics_kernel_v2(int n_nodes, Graph<edge_t>* DAG = NULL);
	static cl_event run_reduce_kernel(cl_int nwg, cl_mem d_output, cl_mem d_input, cl_int npairs, cl_int num_events_to_wait, cl_event* to_wait);
	static cl_event run_reset_kernel(cl_mem d_input, int nels, int default_value, cl_int num_events_to_wait, cl_event* to_wait);
	//standard
	static tuple<cl_event*, cl_int2*> compute_metrics(Graph<int>* DAG, int* entrypoints);
	static tuple<cl_event*, cl_int2*> compute_metrics_rectangular(Graph<int>* DAG, int* entrypoints);
	//vectorized
	static tuple<cl_event*, cl_int2*> compute_metrics_vectorized8_rectangular(Graph<int>* DAG, int* entrypoints);
	static cl_int reduce(int n_nodes, cl_mem to_reduce, cl_int num_events_to_wait, cl_event* to_wait);
	static tuple<cl_event*, cl_int2*> compute_metrics_vectorized_rectangular(Graph<int>* DAG, int* entrypoints);
	static tuple<cl_event*, cl_int2*> compute_metrics_vectorized_v2(Graph<int>* DAG, int* entrypoints);
	static tuple<cl_event*, cl_int2*> compute_metrics_vectorized_v1(Graph<int>* DAG, int* entrypoints);
public:
	/// <summary>
	/// Run the standard version that works with queue of int.
	/// </summary>
	/// <param name="version">
	/// specify the version of the kernel to run, 0 is the default and means the latest version
	/// </param>
	/// <returns>the metrics for the DAG</returns>
	static tuple<cl_event*, cl_int2*> Run(Graph<int>* DAG, int* entrypoints);
	/// <summary>
	/// Run the vectorized version that works with queue of cl_int4.
	/// </summary>
	/// <param name="version">
	/// Specify the version of the kernel to run, 0 is the default and means the latest version
	/// </param>
	/// <returns>the metrics for the DAG</returns>
	static tuple<cl_event*, cl_int2*> RunVectorized(Graph<int>* DAG, int* entrypoints);
};