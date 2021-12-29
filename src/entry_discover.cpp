#include "app_globals.h"
#include "entry_discover.h"
#include "utils.h"
#include "ocl_manager.h"
#include "ocl_buffer_manager.h"
#include <iostream>

tuple<cl_event, int*> EntryDiscover::Run(Graph<int>* DAG) {
	return entry_discover(DAG);
}

tuple<cl_event, int*> EntryDiscover::entry_discover(Graph<int> *DAG) {
	int const n_nodes = DAG->len;

	int* n_entrypoints = new int(0);
	int *entrypoints = new int[n_nodes]; for (int i = 0; i < n_nodes; i++) entrypoints[i] = 0;

	OCLBufferManager BufferManager = *OCLBufferManager::GetInstance();
	OCLManager CLManager = *OCLManager::GetInstance();

	//PASSARE I DATI ALLA GPU
	BufferManager.SetGraphEdges(DAG->adj); //TODO: why here? maybe is better to to this on the main before calling entry discover. It's actually right to allocate memory just when needed but i feel like this shoudn't be here.
	BufferManager.SetNEntrypoints(n_entrypoints);
	BufferManager.SetEntrypoints(entrypoints);

	cl_event entry_discover_evt = run_entry_discover_kernel(n_nodes);

	BufferManager.GetEntrypointsResult(entrypoints, &entry_discover_evt, 1);
	//BufferManager.GetNEntrypointsResult(n_entrypoints, &entry_discover_evt, 1);

	printf("entries discovered\n");
	if (DEBUG_ENTRY_DISCOVER) {
		//cout<<"entrypoints: "<< *n_entrypoints <<endl;
		print(entrypoints, DAG->len, ", ", true);
		cout<<"\n";
	}
	cout << endl;

	//PULIZIA FINALE
	//BufferManager.ReleaseGraphEdges(); //compute metrics is usign it
	BufferManager.ReleaseEntrypoints(); //TODO: i can map queue buffer on entrypoints buffer because they are the same thing but for now i'm letting this way
	BufferManager.ReleaseNEntrypoints();
	CLManager.ReleaseEntryDiscoverKernel();
	return make_tuple(entry_discover_evt, entrypoints);
}


cl_event EntryDiscover::run_entry_discover_kernel(int n_nodes) {
	OCLBufferManager BufferManager = *OCLBufferManager::GetInstance();
	OCLManager CLManager = *OCLManager::GetInstance();

	int arg_index = 0;
	const size_t lws[] = { CLManager.preferred_wg_size };
	const size_t gws[] = { round_mul_up(n_nodes, lws[0]) };

	cl_int err;

	cl_mem graph_edges_GPU = BufferManager.GetGraphEdges();
	cl_mem n_entrypoints_GPU = BufferManager.GetNEntrypoints();
	cl_mem entrypoints_GPU = BufferManager.GetEntrypoints();

	err = clSetKernelArg(CLManager.GetEntryDiscoverKernel(), arg_index++, sizeof(n_nodes), &n_nodes);
	ocl_check(err, "set arg %d for entry_discover_k", arg_index);
	err = clSetKernelArg(CLManager.GetEntryDiscoverKernel(), arg_index++, sizeof(graph_edges_GPU), &graph_edges_GPU);
	ocl_check(err, "set arg %d for entry_discover_k", arg_index);
	err = clSetKernelArg(CLManager.GetEntryDiscoverKernel(), arg_index++, sizeof(n_entrypoints_GPU), &n_entrypoints_GPU);
	ocl_check(err, "set arg %d for entry_discover_k", arg_index);
	err = clSetKernelArg(CLManager.GetEntryDiscoverKernel(), arg_index++, sizeof(entrypoints_GPU), &entrypoints_GPU);
	ocl_check(err, "set arg %d for entry_discover_k", arg_index);

	cl_event entry_discover_evt;
	err = clEnqueueNDRangeKernel(CLManager.queue, 
		CLManager.GetEntryDiscoverKernel(),
		1, NULL, gws, lws,
		0, NULL, &entry_discover_evt);

	ocl_check(err, "enqueue entry_discover kernel");

	return entry_discover_evt;
}