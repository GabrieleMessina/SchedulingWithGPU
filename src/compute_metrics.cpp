#include "compute_metrics.h"
#include "app_globals.h"
#include "utils.h"
#include "ocl_manager.h"
#include "ocl_buffer_manager.h"
#include <iostream>

tuple<cl_event*, cl_int2*> ComputeMetrics::Run(Graph<int>* DAG, int* entrypoints) {
	ComputeMetricsVersion version = OCLManager::compute_metrics_version_chosen;
	switch (version)
	{
		case ComputeMetricsVersion::Latest:
		case ComputeMetricsVersion::Rectangular:
			if (!RECTANGULAR_ADJ) error("set RECTANGULAR_ADJ to true in app_globals to proceed.");
			else return compute_metrics_rectangular(DAG, entrypoints);
		case ComputeMetricsVersion::v1:
		case ComputeMetricsVersion::Working:
		default:
			return compute_metrics(DAG, entrypoints);
	}
}
tuple<cl_event*, cl_int2*> ComputeMetrics::RunVectorized(Graph<int>* DAG, int* entrypoints) {
	VectorizedComputeMetricsVersion version = OCLManager::compute_metrics_vetorized_version_chosen;
	switch (version)
	{
		case VectorizedComputeMetricsVersion::Latest:
		case VectorizedComputeMetricsVersion::RectangularVec8:
			if (!RECTANGULAR_ADJ) error("set RECTANGULAR_ADJ to true in app_globals to proceed.");
			return compute_metrics_vectorized8_rectangular(DAG, entrypoints);
		case VectorizedComputeMetricsVersion::RectangularV2:
		case VectorizedComputeMetricsVersion::Rectangular:
			if (!RECTANGULAR_ADJ) error("set RECTANGULAR_ADJ to true in app_globals to proceed.");
			return compute_metrics_vectorized_rectangular(DAG, entrypoints);
		case VectorizedComputeMetricsVersion::v2:
			return compute_metrics_vectorized_v2(DAG, entrypoints);
		
		case VectorizedComputeMetricsVersion::v1:
			return compute_metrics_vectorized_v1(DAG, entrypoints);

		case VectorizedComputeMetricsVersion::Working:
		default:
			return compute_metrics_vectorized_v1(DAG, entrypoints);
	}
}


tuple<cl_event*, cl_int2*> ComputeMetrics::compute_metrics(Graph<int>* DAG, int *entrypoints) {
	int const n_nodes = DAG->len;
	int* queue = DBG_NEW int[n_nodes]; for (int i = 0; i < n_nodes; i++) queue[i] = entrypoints[i]; //alias for clarity
	int *next_queue = DBG_NEW int[n_nodes]; for (int i = 0; i < n_nodes; i++) next_queue[i] = 0;
	const int metrics_len = GetMetricsArrayLenght(n_nodes); //necessario usare il round alla prossima potenza del due perché altrimenti il sort non potrebbe funzionare
	if (metrics_len < (n_nodes)) error("array metrics più piccolo di nodes array");

	cl_int2 *metrics = DBG_NEW cl_int2[metrics_len];
	int matrixToArrayIndex;
	int parent_of_node;
	for (int i = 0; i < metrics_len; i++) {
		metrics[i].x = 0;
		metrics[i].y = 0;
		if (i > n_nodes) continue;
		for (int j = 0; j < n_nodes; j++) {
			parent_of_node = DAG->hasEdgeByIndex(j, i);
			if (parent_of_node > 0) metrics[i].y++;
		}
	}

	/*cout << "metrics init\n";
	print(metrics, metrics_len, "\n", true);
	cout << "\n";*/

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
		compute_metrics_evt_end = run_compute_metrics_kernel(n_nodes, flip, DAG);
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

tuple<cl_event*, cl_int2*> ComputeMetrics::compute_metrics_rectangular(Graph<int>* DAG, int *entrypoints) {
	int const n_nodes = DAG->len;
	int* queue = DBG_NEW int[n_nodes]; for (int i = 0; i < n_nodes; i++) queue[i] = entrypoints[i]; //alias for clarity
	int *next_queue = DBG_NEW int[n_nodes]; for (int i = 0; i < n_nodes; i++) next_queue[i] = 0;
	const int metrics_len = GetMetricsArrayLenght(n_nodes); //necessario usare il round alla prossima potenza del due perché altrimenti il sort non potrebbe funzionare
	if (metrics_len < (n_nodes)) error("array metrics più piccolo di nodes array");

	cl_int2 *metrics = DBG_NEW cl_int2[metrics_len];
	int matrixToArrayIndex;
	int parent_of_node;
	for (int i = 0; i < metrics_len; i++) {
		metrics[i].x = 0;
		metrics[i].y = 0;
		if (i > n_nodes) continue;
		for (int j = 0; j < n_nodes; j++) {
			parent_of_node = DAG->hasEdgeByIndex(j, i);
			if (parent_of_node > 0) metrics[i].y++;
		}
	}

	/*cout << "metrics init\n";
	print(metrics, metrics_len, "\n", true);
	cout << "\n";*/

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
		compute_metrics_evt_end = run_compute_metrics_kernel(n_nodes, flip, DAG);
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
	BufferManager.ReleaseGraphReverseEdges();
	//BufferManager.ReleaseMetrics(); //sort kernel is using it

	delete[] queue;
	delete[] next_queue;

	cl_event* compute_metrics_evt = DBG_NEW cl_event[2];
	compute_metrics_evt[0] = compute_metrics_evt_start;
	compute_metrics_evt[1] = compute_metrics_evt_end;
	return make_tuple(compute_metrics_evt, metrics);
}


tuple<cl_event*, cl_int2*> ComputeMetrics::compute_metrics_vectorized_v1(Graph<int>* DAG, int* entrypoints) {
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
	int matrixToArrayIndex;
	for (int i = 0; i < metrics_len; i++) {
		metrics[i].x = 0;
		metrics[i].y = 0;
		if (i > n_nodes) continue;
		for (int j = 0; j < n_nodes; j++) {
			int parent_of_node = DAG->hasEdgeByIndex(i, j);
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

tuple<cl_event*, cl_int2*> ComputeMetrics::compute_metrics_vectorized_v2(Graph<int>* DAG, int* entrypoints) {
	int const n_nodes = DAG->len;
	int const queue_len = ceil(n_nodes / 4.0);
	const int metrics_len = GetMetricsArrayLenght(n_nodes); //necessario usare il round alla prossima potenza del due perché altrimenti il sort non potrebbe funzionare
	if (metrics_len < (n_nodes)) error("array metrics più piccolo di nodes array");

	cl_int4* queue = DBG_NEW cl_int4[queue_len];
	
	for (int i = 0; i < queue_len; i++) {
		queue[i].x = 0;
		queue[i].y = 0;
		queue[i].z = 0;
		queue[i].w = 0;

		for (int k = 0; k < n_nodes; k++)
		{
			int j = i * 4;
			//il valore iniziale in coda per ogni task è dato dal numero di parent da cui dipende e che deve aspettare prima di poter essere eseguito
			//TODO: a questo punto la ricerca degli entry point è inutile perchè sto già mettendoli a zero in queue.
			queue[i].x += (j < n_nodes && DAG->hasEdgeByIndex(k, j++) > 0) ? 1 : 0;
			queue[i].y += (j < n_nodes && DAG->hasEdgeByIndex(k, j++) > 0) ? 1 : 0;
			queue[i].z += (j < n_nodes && DAG->hasEdgeByIndex(k, j++) > 0) ? 1 : 0;
			queue[i].w += (j < n_nodes && DAG->hasEdgeByIndex(k, j)   > 0) ? 1 : 0;
		}
	}

	/*cout << "queue init\n";
	print(queue, queue_len, "\n", true);
	cout << "\n";*/

	cl_int2* metrics = DBG_NEW cl_int2[metrics_len]; for (int i = 0; i < metrics_len; i++) metrics[i] = cl_int2{ DAG->nodes[i],0};

	OCLBufferManager BufferManager = *OCLBufferManager::GetInstance();

	BufferManager.SetNodes(DAG->nodes);
	BufferManager.SetQueue(queue);
	BufferManager.SetMetrics(metrics);

	//ESEGUIRE L'ALGORITMO DI SCHEDULING SU GPU
	cl_event compute_metrics_evt_start = cl_event(), compute_metrics_evt_end = cl_event();

	bool moreToProcess = false; 
	int count = 0;
	do
	{
		compute_metrics_evt_end = run_compute_metrics_kernel_v2(n_nodes);
		if(count == 0) compute_metrics_evt_start = compute_metrics_evt_end;

		BufferManager.GetQueueResult(queue, &compute_metrics_evt_end, 1);

		moreToProcess = !isEmpty(queue, queue_len, cl_int4{ -1,-1,-1,-1 });
		count++;
	} while (moreToProcess);

	BufferManager.GetMetricsResult(metrics, &compute_metrics_evt_end, 1);
	printf("metrics computed\n");
	if (DEBUG_COMPUTE_METRICS) {
		cout << "metrics: " << metrics_len << endl;
		cout << "count:" << count << endl;
		print(metrics, metrics_len, "\n", true);
		cout << "\n";
	}

	//PULIZIA FINALE
	BufferManager.ReleaseNodes();
	BufferManager.ReleaseQueue();
	BufferManager.ReleaseNextQueue();
	BufferManager.ReleaseGraphEdges();
	//BufferManager.ReleaseMetrics(); //sort kernel is using it

	delete[] queue;
	//delete[] next_queue;

	cl_event* compute_metrics_evt = DBG_NEW cl_event[2];
	compute_metrics_evt[0] = compute_metrics_evt_start;
	compute_metrics_evt[1] = compute_metrics_evt_end;
	return make_tuple(compute_metrics_evt, metrics);
}

tuple<cl_event*, cl_int2*> ComputeMetrics::compute_metrics_vectorized_rectangular(Graph<int>* DAG, int* entrypoints) {
	int const n_nodes = DAG->len;
	int queue_len = ceil(n_nodes / 4.0);
	const int metrics_len = GetMetricsArrayLenght(n_nodes); //necessario usare il round alla prossima potenza del due perché altrimenti il sort non potrebbe funzionare
	if (metrics_len < (n_nodes)) error("array metrics più piccolo di nodes array");

	cl_int4* queue = DBG_NEW cl_int4[queue_len];

	for (int i = 0; i < queue_len; i++) {
		int j = i * 4;
		queue[i].x = DAG->numberOfParentOfNode(j++);
		queue[i].y = DAG->numberOfParentOfNode(j++);
		queue[i].z = DAG->numberOfParentOfNode(j++);
		queue[i].w = DAG->numberOfParentOfNode(j);
	}

	/*cout << "queue init\n";
	print(queue, queue_len, "\n", true);
	cout << "\n";*/

	cl_int2* metrics = DBG_NEW cl_int2[metrics_len]; for (int i = 0; i < metrics_len; i++) metrics[i] = cl_int2{ DAG->nodes[i],0 };

	OCLBufferManager BufferManager = *OCLBufferManager::GetInstance();

	BufferManager.SetNodes(DAG->nodes);
	BufferManager.SetQueue(queue);
	BufferManager.SetMetrics(metrics);

	//ESEGUIRE L'ALGORITMO DI SCHEDULING SU GPU
	cl_event compute_metrics_evt_start = cl_event(), compute_metrics_evt_end = cl_event();

	bool moreToProcess = false;
	int count = 0;
	do
	{
		compute_metrics_evt_end = run_compute_metrics_kernel_v2(n_nodes, DAG);
		if (count == 0) compute_metrics_evt_start = compute_metrics_evt_end;

		BufferManager.GetQueueResult(queue, &compute_metrics_evt_end, 1);

		moreToProcess = !isEmpty(queue, queue_len, cl_int4{ -1,-1,-1,-1 });
		count++;
	} while (moreToProcess);

	BufferManager.GetMetricsResult(metrics, &compute_metrics_evt_end, 1);
	printf("metrics computed\n");
	if (DEBUG_COMPUTE_METRICS) {
		cout << "metrics: " << metrics_len << endl;
		cout << "count:" << count << endl;
		print(metrics, metrics_len, "\n", true);
		cout << "\n";
	}

	//PULIZIA FINALE
	BufferManager.ReleaseNodes();
	BufferManager.ReleaseQueue();
	BufferManager.ReleaseNextQueue();
	BufferManager.ReleaseGraphEdges();
	//BufferManager.ReleaseMetrics(); //sort kernel is using it

	delete[] queue;
	//delete[] next_queue;

	cl_event* compute_metrics_evt = DBG_NEW cl_event[2];
	compute_metrics_evt[0] = compute_metrics_evt_start;
	compute_metrics_evt[1] = compute_metrics_evt_end;
	return make_tuple(compute_metrics_evt, metrics);
}


tuple<cl_event*, cl_int2*> ComputeMetrics::compute_metrics_vectorized8_rectangular(Graph<int>* DAG, int* entrypoints) {
	int const n_nodes = DAG->len;
	int queue_len = ceil(n_nodes / 8.0);
	const int metrics_len = GetMetricsArrayLenght(n_nodes); //necessario usare il round alla prossima potenza del due perché altrimenti il sort non potrebbe funzionare
	if (metrics_len < (n_nodes)) error("array metrics più piccolo di nodes array");

	cl_int8* queue = DBG_NEW cl_int8[queue_len];

	for (int i = 0; i < queue_len; i++) {
		int j = i * 8;
		queue[i].s0 = DAG->numberOfParentOfNode(j++);
		queue[i].s1 = DAG->numberOfParentOfNode(j++);
		queue[i].s2 = DAG->numberOfParentOfNode(j++);
		queue[i].s3 = DAG->numberOfParentOfNode(j++);
		queue[i].s4 = DAG->numberOfParentOfNode(j++);
		queue[i].s5 = DAG->numberOfParentOfNode(j++);
		queue[i].s6 = DAG->numberOfParentOfNode(j++);
		queue[i].s7 = DAG->numberOfParentOfNode(j);
	}

	/*cout << "queue init\n";
	print(queue, queue_len, "\n", true);
	cout << "\n";*/

	cl_int2* metrics = DBG_NEW cl_int2[metrics_len]; for (int i = 0; i < metrics_len; i++) metrics[i] = cl_int2{ DAG->nodes[i],0 };

	OCLBufferManager BufferManager = *OCLBufferManager::GetInstance();

	BufferManager.SetNodes(DAG->nodes);
	BufferManager.SetQueue(queue);
	BufferManager.SetMetrics(metrics);

	//ESEGUIRE L'ALGORITMO DI SCHEDULING SU GPU
	cl_event compute_metrics_evt_start = cl_event(), compute_metrics_evt_end = cl_event();

	bool moreToProcess = false;
	int count = 0;
	do
	{
		compute_metrics_evt_end = run_compute_metrics_kernel_v2(n_nodes, DAG);
		if (count == 0) compute_metrics_evt_start = compute_metrics_evt_end;

		BufferManager.GetQueueResult(queue, &compute_metrics_evt_end, 1);

		moreToProcess = !isEmpty(queue, queue_len, cl_int8{ -1,-1,-1,-1,-1,-1,-1,-1 });
		count++;
	} while (moreToProcess);

	BufferManager.GetMetricsResult(metrics, &compute_metrics_evt_end, 1);
	printf("metrics computed\n");
	if (DEBUG_COMPUTE_METRICS) {
		cout << "metrics: " << metrics_len << endl;
		cout << "count:" << count << endl;
		print(metrics, metrics_len, "\n", true);
		cout << "\n";
	}

	//PULIZIA FINALE
	BufferManager.ReleaseNodes();
	BufferManager.ReleaseQueue();
	BufferManager.ReleaseNextQueue();
	BufferManager.ReleaseGraphEdges();
	//BufferManager.ReleaseMetrics(); //sort kernel is using it

	delete[] queue;
	//delete[] next_queue;

	cl_event* compute_metrics_evt = DBG_NEW cl_event[2];
	compute_metrics_evt[0] = compute_metrics_evt_start;
	compute_metrics_evt[1] = compute_metrics_evt_end;
	return make_tuple(compute_metrics_evt, metrics);
}


cl_event ComputeMetrics::run_compute_metrics_kernel(int n_nodes, bool flip, Graph<edge_t>* DAG) {
	//TODO: queste variabili potrebbero essere statiche
	OCLBufferManager BufferManager = *OCLBufferManager::GetInstance();

	int arg_index = 0;
	const size_t lws[] = { OCLManager::preferred_wg_size };
	const size_t gws[] = { round_mul_up(n_nodes, lws[0]) };

	cl_mem graph_nodes_GPU = BufferManager.GetNodes();
	cl_mem queue_GPU = (flip) ? BufferManager.GetNextQueue() : BufferManager.GetQueue();
	cl_mem next_queue_GPU = (flip) ? BufferManager.GetQueue() : BufferManager.GetNextQueue();
	cl_mem graph_edges_GPU = BufferManager.GetGraphEdges();
	cl_mem graph_edges_reverse_GPU = BufferManager.GetGraphReverseEdges();
	cl_mem metrics_GPU = BufferManager.GetMetrics();

	cl_int err;
	ComputeMetricsVersion version = OCLManager::compute_metrics_version_chosen;
	
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
	if (version == ComputeMetricsVersion::Rectangular && DAG != NULL) {
		err = clSetKernelArg(OCLManager::GetComputeMetricsKernel(), arg_index++, sizeof(graph_edges_reverse_GPU), &graph_edges_reverse_GPU);
		ocl_check(err, "set arg %d for compute_metrics_k", arg_index);
	}
	err = clSetKernelArg(OCLManager::GetComputeMetricsKernel(), arg_index++, sizeof(metrics_GPU), &metrics_GPU);
	ocl_check(err, "set arg %d for compute_metrics_k", arg_index);

	if (version == ComputeMetricsVersion::Rectangular && DAG != NULL) {
		int max_adj_dept = DAG->max_edges_for_node;
		err = clSetKernelArg(OCLManager::GetComputeMetricsKernel(), arg_index++, sizeof(max_adj_dept), &max_adj_dept);
		ocl_check(err, "set arg %d for compute_metrics_k", arg_index);
	}

	cl_event compute_metrics_evt;
	err = clEnqueueNDRangeKernel(OCLManager::queue,
		OCLManager::GetComputeMetricsKernel(),
		1, NULL, gws, lws,
		0, NULL, &compute_metrics_evt);

	ocl_check(err, "enqueue compute_metrics kernel");

	return compute_metrics_evt;
}


cl_event ComputeMetrics::run_compute_metrics_kernel_v2(int n_nodes, Graph<edge_t>* DAG) {
	OCLBufferManager BufferManager = *OCLBufferManager::GetInstance();
	VectorizedComputeMetricsVersion version = OCLManager::compute_metrics_vetorized_version_chosen;

	int arg_index = 0;
	const size_t lws[] = { OCLManager::preferred_wg_size };
	size_t gws[] = { round_mul_up(n_nodes/4, lws[0]) };
	int local_queue_memsize = lws[0] * sizeof(cl_int4); //TODO: min(preferred e n_nodes/4);

	if (version == VectorizedComputeMetricsVersion::RectangularVec8) {
		gws[0] = round_mul_up(n_nodes / 8, lws[0]);
		local_queue_memsize = lws[0] * sizeof(cl_int8);
	}

	cl_mem graph_nodes_GPU = BufferManager.GetNodes();
	cl_mem queue_GPU = BufferManager.GetQueue();
	cl_mem graph_edges_GPU = BufferManager.GetGraphEdges();
	cl_mem graph_edges_reverse_GPU = BufferManager.GetGraphReverseEdges();
	cl_mem metrics_GPU = BufferManager.GetMetrics();

	cl_int err;

	err = clSetKernelArg(OCLManager::GetComputeMetricsKernel(), arg_index++, sizeof(graph_nodes_GPU), &graph_nodes_GPU);
	ocl_check(err, "set arg %d for compute_metrics_k", arg_index);
	err = clSetKernelArg(OCLManager::GetComputeMetricsKernel(), arg_index++, sizeof(queue_GPU), &queue_GPU);
	ocl_check(err, "set arg %d for compute_metrics_k", arg_index);
	if (version != VectorizedComputeMetricsVersion::RectangularV2 && version != VectorizedComputeMetricsVersion::RectangularVec8) {
		err = clSetKernelArg(OCLManager::GetComputeMetricsKernel(), arg_index++, local_queue_memsize, NULL);
		ocl_check(err, "set arg %d for compute_metrics_k", arg_index);
		err = clSetKernelArg(OCLManager::GetComputeMetricsKernel(), arg_index++, local_queue_memsize, NULL);
		ocl_check(err, "set arg %d for compute_metrics_k", arg_index);
	}
	err = clSetKernelArg(OCLManager::GetComputeMetricsKernel(), arg_index++, sizeof(n_nodes), &n_nodes);
	ocl_check(err, "set arg %d for compute_metrics_k", arg_index);
	err = clSetKernelArg(OCLManager::GetComputeMetricsKernel(), arg_index++, sizeof(graph_edges_GPU), &graph_edges_GPU);
	ocl_check(err, "set arg %d for compute_metrics_k", arg_index);
	if ((version == VectorizedComputeMetricsVersion::Rectangular || version == VectorizedComputeMetricsVersion::RectangularV2 || version == VectorizedComputeMetricsVersion::RectangularVec8) && DAG != NULL) {
		err = clSetKernelArg(OCLManager::GetComputeMetricsKernel(), arg_index++, sizeof(graph_edges_reverse_GPU), &graph_edges_reverse_GPU);
		ocl_check(err, "set arg %d for compute_metrics_k", arg_index);
	}
	
	err = clSetKernelArg(OCLManager::GetComputeMetricsKernel(), arg_index++, sizeof(metrics_GPU), &metrics_GPU);
	ocl_check(err, "set arg %d for compute_metrics_k", arg_index);

	if ((version == VectorizedComputeMetricsVersion::Rectangular || version == VectorizedComputeMetricsVersion::RectangularV2 || version == VectorizedComputeMetricsVersion::RectangularVec8) && DAG != NULL) {
		int max_adj_dept = DAG->max_edges_for_node;
		err = clSetKernelArg(OCLManager::GetComputeMetricsKernel(), arg_index++, sizeof(max_adj_dept), &max_adj_dept);
		ocl_check(err, "set arg %d for compute_metrics_k", arg_index);
	}

	cl_event compute_metrics_evt;
	err = clEnqueueNDRangeKernel(OCLManager::queue,
		OCLManager::GetComputeMetricsKernel(),
		1, NULL, gws, lws,
		0, NULL, &compute_metrics_evt);

	ocl_check(err, "enqueue compute_metrics kernel");

	return compute_metrics_evt;
}