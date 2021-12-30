#include "compute_metrics.h"
#include "app_globals.h"
#include "utils.h"
#include "ocl_manager.h"
#include "ocl_buffer_manager.h"
#include <iostream>

//TODO: ottimizzare, provare ad eseguire tutti i passaggi in GPU.

tuple<cl_event*, cl_int2*> ComputeMetrics::Run(Graph<int>* DAG, int* entrypoints, int version) {
	switch (version)
	{
		case 1:
			return compute_metrics(DAG, entrypoints);
		default:
			return compute_metrics(DAG, entrypoints);
	}
}
tuple<cl_event*, cl_int2*> ComputeMetrics::RunVectorized(Graph<int>* DAG, int* entrypoints, int version) {
	switch (version)
	{
		case 1:
			return compute_metrics_vectorized(DAG, entrypoints);
		default:
			return compute_metrics_vectorized(DAG, entrypoints);
	}
}


tuple<cl_event*, cl_int2*> ComputeMetrics::compute_metrics(Graph<int>* DAG, int *entrypoints) {
	int const n_nodes = DAG->len;
	int* queue = DBG_NEW int[n_nodes]; for (int i = 0; i < n_nodes; i++) queue[i] = entrypoints[i]; //alias for clarity
	int *next_queue = DBG_NEW int[n_nodes]; for (int i = 0; i < n_nodes; i++) next_queue[i] = 0;
	const int metrics_len = GetMetricsArrayLenght(n_nodes); //necessario usare il round alla prossima potenza del due perché altrimenti il sort non potrebbe funzionare
	if (metrics_len < (n_nodes)) error("array metrics più piccolo di nodes array");

	cl_int2 *metrics = DBG_NEW cl_int2[metrics_len];
	for (int i = 0; i < metrics_len; i++) {
		metrics[i].x = 0;
		metrics[i].y = 0;
		if (i > n_nodes) continue;
		for (int j = 0; j < n_nodes; j++) {
			int parent_of_node = DAG->adj[matrix_to_array_indexes(j, i, n_nodes)];
			if (parent_of_node > 0) metrics[i].y++;
		}
	}

	OCLBufferManager BufferManager = *OCLBufferManager::GetInstance();

	BufferManager.SetNodes(DAG->nodes);
	BufferManager.SetQueue(queue);
	BufferManager.SetNextQueue(next_queue);
	BufferManager.SetMetrics(metrics);

	//ESEGUIRE L'ALGORITMO DI SCHEDULING SU GPU
	cl_event compute_metrics_evt_start = cl_event(), compute_metrics_evt_end = cl_event();

	int count = 0;
	bool flip = false;
	bool moreToProcess = true;
	do {
		//scambio le due code e reinizializzo la next_queue in modo da poter essere riutilizzata.
		compute_metrics_evt_end = run_compute_metrics_kernel(n_nodes, flip);
		if (count == 0) compute_metrics_evt_start = compute_metrics_evt_end;
		
		if(flip)
			BufferManager.GetQueueResult(next_queue, &compute_metrics_evt_end, 1);
		else
			BufferManager.GetNextQueueResult(next_queue, &compute_metrics_evt_end, 1);

		moreToProcess = !isEmpty(next_queue, n_nodes, 0);

		/*cout << "next queue\n";
		print(next_queue, DAG->len, "\n", true);
		cout << "\n";*/

		for (int i = 0; i < n_nodes; i++) next_queue[i] = 0;
		
		//passo il next_array reinizializzato alla GPU.
		if (!flip)
			BufferManager.SetQueue(next_queue);
		else
			BufferManager.SetNextQueue(next_queue);

		flip = !flip;
		count++;
	} while (moreToProcess);
	
	BufferManager.GetMetricsResult(metrics, &compute_metrics_evt_end, 1);
	
	cout << "count:" << count << endl;
	printf("metrics computed\n");
	if (DEBUG_COMPUTE_METRICS) {
		cout<<"metrics: "<<metrics_len<<endl;
		print(metrics, DAG->len, "\n", true);
		cout<<"\n";
	}
	cout << endl;

	//PULIZIA FINALE
	BufferManager.ReleaseNodes();
	BufferManager.ReleaseQueue();
	BufferManager.ReleaseNextQueue();
	BufferManager.ReleaseGraphEdges();
	//BufferManager.ReleaseMetrics(); //sort kernel is using it

	delete[] queue;
	delete[] next_queue;

	cl_event* compute_metrics_evt = DBG_NEW cl_event[2];
	compute_metrics_evt[0] = compute_metrics_evt_start;
	compute_metrics_evt[1] = compute_metrics_evt_end;
	return make_tuple(compute_metrics_evt, metrics);
}


tuple<cl_event*, cl_int2*> ComputeMetrics::compute_metrics_vectorized(Graph<int>* DAG, int* entrypoints) {
	int const n_nodes = DAG->len;
	int const queue_len = ceil(n_nodes / 4.0);
	const int metrics_len = GetMetricsArrayLenght(n_nodes); //necessario usare il round alla prossima potenza del due perché altrimenti il sort non potrebbe funzionare
	if (metrics_len < (n_nodes)) error("array metrics più piccolo di nodes array");
	
	cl_int4* queue = DBG_NEW cl_int4[queue_len];
	cl_int4* next_queue = DBG_NEW cl_int4[queue_len]; for (int i = 0; i < queue_len; i++) next_queue[i] = cl_int4{ 0,0,0,0 };
	for (int i = 0; i < queue_len; i++) {
		int j = i * 4;
		queue[i].x = (j < n_nodes) ? entrypoints[j++] : 0;
		queue[i].y = (j < n_nodes) ? entrypoints[j++] : 0;
		queue[i].z = (j < n_nodes) ? entrypoints[j++] : 0;
		queue[i].w = (j < n_nodes) ? entrypoints[j] : 0;
	}

	cl_int2* metrics = DBG_NEW cl_int2[metrics_len];
	for (int i = 0; i < metrics_len; i++) {
		metrics[i].x = 0;
		metrics[i].y = 0;
		if (i > n_nodes) continue;
		for (int j = 0; j < n_nodes; j++) {
			int parent_of_node = DAG->adj[matrix_to_array_indexes(j, i, n_nodes)];
			if (parent_of_node > 0) metrics[i].y++;
		}
	}

	OCLBufferManager BufferManager = *OCLBufferManager::GetInstance();

	BufferManager.SetNodes(DAG->nodes);
	BufferManager.SetQueue(queue);
	BufferManager.SetNextQueue(next_queue);
	BufferManager.SetMetrics(metrics);

	//ESEGUIRE L'ALGORITMO DI SCHEDULING SU GPU
	cl_event compute_metrics_evt_start = cl_event(), compute_metrics_evt_end = cl_event();

	int count = 0;
	bool flip = false;
	bool moreToProcess = true;
	do {
		//scambio le due code e reinizializzo la next_queue in modo da poter essere riutilizzata.
		compute_metrics_evt_end = run_compute_metrics_kernel(n_nodes, flip);
		if (count == 0) compute_metrics_evt_start = compute_metrics_evt_end;

		if (flip)
			BufferManager.GetQueueResult(next_queue, &compute_metrics_evt_end, 1);
		else
			BufferManager.GetNextQueueResult(next_queue, &compute_metrics_evt_end, 1);

		moreToProcess = !isEmpty(next_queue, queue_len, cl_int4{ 0,0,0,0 });

		/*cout << "next queue\n";
		print(next_queue, DAG->len, "\n", true);
		cout << "\n";*/

		for (int i = 0; i < queue_len; i++) next_queue[i] = cl_int4{ 0,0,0,0 };

		//passo il next_array reinizializzato alla GPU.
		if (!flip)
			BufferManager.SetQueue(next_queue);
		else
			BufferManager.SetNextQueue(next_queue);

		flip = !flip;
		count++;
	} while (moreToProcess);

	BufferManager.GetMetricsResult(metrics, &compute_metrics_evt_end, 1);

	cout << "count:" << count << endl;
	printf("metrics computed\n");
	if (DEBUG_COMPUTE_METRICS) {
		cout << "metrics: " << metrics_len << endl;
		print(metrics, DAG->len, "\n", true);
		cout << "\n";
	}

	//PULIZIA FINALE
	BufferManager.ReleaseNodes();
	BufferManager.ReleaseQueue();
	BufferManager.ReleaseNextQueue();
	BufferManager.ReleaseGraphEdges();
	//BufferManager.ReleaseMetrics(); //sort kernel is using it

	delete[] queue;
	delete[] next_queue;

	cl_event* compute_metrics_evt = DBG_NEW cl_event[2];
	compute_metrics_evt[0] = compute_metrics_evt_start;
	compute_metrics_evt[1] = compute_metrics_evt_end;
	return make_tuple(compute_metrics_evt, metrics);
}


cl_event ComputeMetrics::run_compute_metrics_kernel(int n_nodes, bool flip) {
	OCLBufferManager BufferManager = *OCLBufferManager::GetInstance();

	int arg_index = 0;
	const size_t lws[] = { OCLManager::preferred_wg_size };
	const size_t gws[] = { round_mul_up(n_nodes, lws[0]) };

	cl_mem graph_nodes_GPU = BufferManager.GetNodes();
	cl_mem queue_GPU = (flip) ? BufferManager.GetNextQueue() : BufferManager.GetQueue();
	cl_mem next_queue_GPU = (flip) ? BufferManager.GetQueue() : BufferManager.GetNextQueue();
	cl_mem graph_edges_GPU = BufferManager.GetGraphEdges();
	cl_mem metrics_GPU = BufferManager.GetMetrics();

	cl_int err;
	err = clSetKernelArg(OCLManager::GetComputeMetricsKernel(), arg_index++, sizeof(graph_nodes_GPU), &graph_nodes_GPU);
	ocl_check(err, "set arg %d for compute_metrics_k", arg_index);
	err = clSetKernelArg(OCLManager::GetComputeMetricsKernel(), arg_index++, sizeof(queue_GPU), &queue_GPU);
	ocl_check(err, "set arg %d for compute_metrics_k", arg_index);
	err = clSetKernelArg(OCLManager::GetComputeMetricsKernel(), arg_index++, sizeof(next_queue_GPU), &next_queue_GPU);
	ocl_check(err, "set arg %d for compute_metrics_k", arg_index);
	err = clSetKernelArg(OCLManager::GetComputeMetricsKernel(), arg_index++, sizeof(n_nodes), &n_nodes);
	ocl_check(err, "set arg %d for compute_metrics_k", arg_index);
	err = clSetKernelArg(OCLManager::GetComputeMetricsKernel(), arg_index++, sizeof(graph_edges_GPU), &graph_edges_GPU);
	ocl_check(err, "set arg %d for compute_metrics_k", arg_index);
	err = clSetKernelArg(OCLManager::GetComputeMetricsKernel(), arg_index++, sizeof(metrics_GPU), &metrics_GPU);
	ocl_check(err, "set arg %d for compute_metrics_k", arg_index);

	cl_event compute_metrics_evt;
	err = clEnqueueNDRangeKernel(OCLManager::queue,
		OCLManager::GetComputeMetricsKernel(),
		1, NULL, gws, lws,
		0, NULL, &compute_metrics_evt);

	ocl_check(err, "enqueue compute_metrics kernel");

	return compute_metrics_evt;
}