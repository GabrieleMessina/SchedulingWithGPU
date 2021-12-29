#pragma once
#include "dag.h"
#include "ocl_boiler.h"
#include <tuple>


class ComputeMetrics {
private:
	static cl_event run_compute_metrics_kernel(int n_nodes, bool flip);
	static tuple<cl_event*, cl_int2*> compute_metrics(Graph<int>* DAG, int* entrypoints);
	static tuple<cl_event*, cl_int2*> compute_metrics_vectorized(Graph<int>* DAG, int* entrypoints);
public:
	/// <summary>
	/// Run the standard version that works with queue of int.
	/// </summary>
	/// <param name="version">
	/// specify the version of the kernel to run, 0 is the default and means the latest version
	/// </param>
	/// <returns>the metrics for the DAG</returns>
	static tuple<cl_event*, cl_int2*> Run(Graph<int>* DAG, int* entrypoints, int version = 0);
	/// <summary>
	/// Run the vectorized version that works with queue of cl_int4.
	/// </summary>
	/// <param name="version">
	/// Specify the version of the kernel to run, 0 is the default and means the latest version
	/// </param>
	/// <returns>the metrics for the DAG</returns>
	static tuple<cl_event*, cl_int2*> RunVectorized(Graph<int>* DAG, int* entrypoints, int version = 0);;
};