//#include "ocl_boiler.h"
//#include "dag.h"
//
//cl_event entry_discover(Graph<int> *DAG, int* entrypoints) {
//	int const n_nodes = DAG->len;
//	size_t nodes_memsize = n_nodes * sizeof(int);
//	size_t edges_memsize = n_nodes * n_nodes * sizeof(bool);
//
//	int* n_entrypoints = new int(0);
//	entrypoints = new int[n_nodes]; for (int i = 0; i < n_nodes; i++) entrypoints[i] = 0;
//	size_t entrypoints_memsize = n_nodes * sizeof(int);
//
//	//CREO MEMORIA BUFFER IN GPU
//	graph_edges_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, edges_memsize, NULL, &err);
//	ocl_check(err, "create buffer graph_edges_GPU");
//	n_entrypoints_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(n_entrypoints), NULL, &err);
//	ocl_check(err, "create buffer n_entrypoints_GPU");
//	entrypoints_GPU = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, entrypoints_memsize, NULL, &err);
//	ocl_check(err, "create buffer entrypoints_GPU");
//
//	//PASSARE I DATI DELLA DAG ALLA GPU
//	cl_event write_nodes_evt, write_edges_evt, write_nentries_evt, write_entries_evt;
//	err = clEnqueueWriteBuffer(que, graph_edges_GPU, CL_TRUE,
//		0, edges_memsize, DAG->adj,
//		0, NULL, &write_edges_evt);
//	ocl_check(err, "write dataset edges into graph_edges_GPU");
//	err = clEnqueueWriteBuffer(que, n_entrypoints_GPU, CL_TRUE,
//		0, sizeof(int), n_entrypoints,
//		0, NULL, &write_nentries_evt);
//	ocl_check(err, "write n_entries into n_entrypoints_GPU");
//	err = clEnqueueWriteBuffer(que, entrypoints_GPU, CL_TRUE,
//		0, entrypoints_memsize, entrypoints,
//		0, NULL, &write_entries_evt);
//	ocl_check(err, "write n_entries into entrypoints_GPU");
//
//	//ESEGUIRE L'ALGORITMO DI SCHEDULING SU GPU
//	cl_event entry_discover_evt = _entry_discover(n_nodes, graph_edges_GPU, n_entrypoints_GPU, entrypoints_GPU);
//
//	//PASSA I RISULTATI DELLO SCHEDULING DALLA GPU ALLA CPU
//	cl_event read_entries_evt;
//	err = clEnqueueReadBuffer(que, entrypoints_GPU, CL_TRUE,
//		0, entrypoints_memsize, entrypoints,
//		1, &entry_discover_evt, &read_entries_evt);
//	ocl_check(err, "read buffer entrypoints_GPU");
//	cl_event read_nentries_evt;
//	err = clEnqueueReadBuffer(que, n_entrypoints_GPU, CL_TRUE,
//		0, sizeof(int), n_entrypoints,
//		1, &entry_discover_evt, &read_nentries_evt);
//	ocl_check(err, "read buffer n_entrypoints_GPU");
//
//	//PULIZIA FINALE
//	clReleaseKernel(entry_discover_k);
//	return entry_discover_evt;
//}