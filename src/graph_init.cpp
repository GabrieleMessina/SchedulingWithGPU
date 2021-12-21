#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <chrono>
#include <ctime>   
#include "dag.h"
#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"
#include "utils/print_stuff.h"

#include "windows.h"
#include "psapi.h"

using namespace std;

void error(char const *str)
{
	fprintf(stderr, "%s\n", str);
	exit(1);
}

bool isEmpty(cl_int4 *v, int len, cl_int4 default_v = cl_int4{0,0,0,0}){
	for (int i = 0; i < len; i++)
	{
		if(v[i].s0 != default_v.s0 || v[i].s1 != default_v.s1 || v[i].s2 != default_v.s2 || v[i].s3 != default_v.s3)
			return false;
	}
	return true;
}

void printInt4(cl_int4 *v, int len, string separator = " ", bool withIndexes = false){
	for (int i = 0; i < len; i++)
	{
		if(withIndexes)
			cout << i << ":";
		cout<<"("<<v[i].x<<", "<<v[i].y<<", "<<v[i].z<<", "<<v[i].w<<")"<<separator;
	}
	
	cout<<"\n";
}

cl_context ctx;
cl_command_queue que;
cl_int err;
cl_kernel entry_discover_k, compute_metrics_k, m_MergesortGlobalBigKernel, m_MergesortGlobalSmallKernel, m_MergesortStartKernel;
void ocl_init(char *progName, char *kernelNameEntryDiscover, char *kernelNameComputeMetrics);
size_t preferred_wg_size;

Graph<int> *DAG;
Graph<int>* initDagWithDataSet(string fileName);

void printMemoryUsage();
void measurePerformance(cl_event entry_discover_evt,cl_event *compute_metrics_evt, cl_event *sort_task_evts, int nels);
void measurePerformance(cl_event evt, int nels, string event_name);
void verify();

int *entrypoints; 
size_t m_N_padded;
cl_event _entry_discover(int n_nodes, cl_mem graph_edges_GPU, cl_mem n_entrypoints_GPU, cl_mem entrypoints_GPU)
{
	int arg_index = 0;
	const size_t lws[] = { preferred_wg_size };
	const size_t gws[] = { round_mul_up(n_nodes, lws[0]) };

	cl_int err;
	err = clSetKernelArg(entry_discover_k, arg_index++, sizeof(n_nodes), &n_nodes);
	ocl_check(err,  "set arg %d for entry_discover_k", arg_index);
	err = clSetKernelArg(entry_discover_k, arg_index++, sizeof(graph_edges_GPU), &graph_edges_GPU);
	ocl_check(err,  "set arg %d for entry_discover_k", arg_index);
	err = clSetKernelArg(entry_discover_k, arg_index++, sizeof(n_entrypoints_GPU), &n_entrypoints_GPU);
	ocl_check(err,  "set arg %d for entry_discover_k", arg_index);
	err = clSetKernelArg(entry_discover_k, arg_index++, sizeof(entrypoints_GPU), &entrypoints_GPU);
	ocl_check(err,  "set arg %d for entry_discover_k", arg_index);

	cl_event entry_discover_evt;
	err = clEnqueueNDRangeKernel(que, entry_discover_k,
		1, NULL, gws, lws,
		0, NULL, &entry_discover_evt);

	ocl_check(err, "enqueue entry_discover");

	return entry_discover_evt;
}

cl_mem graph_edges_GPU;
cl_mem n_entrypoints_GPU;
cl_mem entrypoints_GPU;


cl_event entry_discover(){
	int const n_nodes = DAG->len;
	size_t nodes_memsize = n_nodes*sizeof(int);
	size_t edges_memsize = n_nodes*n_nodes*sizeof(bool);

	int *n_entrypoints = new int(0);
	entrypoints = new int[n_nodes]; for (int i = 0; i < n_nodes; i++) entrypoints[i] = 0;
	size_t entrypoints_memsize = n_nodes*sizeof(int);

	//CREO MEMORIA BUFFER IN GPU
	graph_edges_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, edges_memsize, NULL, &err);
	ocl_check(err, "create buffer graph_edges_GPU");
	n_entrypoints_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(n_entrypoints), NULL, &err);
	ocl_check(err, "create buffer n_entrypoints_GPU");
	entrypoints_GPU = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, entrypoints_memsize, NULL, &err);
	ocl_check(err, "create buffer entrypoints_GPU");

	//PASSARE I DATI DELLA DAG ALLA GPU 
	cl_event write_nodes_evt, write_edges_evt, write_nentries_evt, write_entries_evt;
	err = clEnqueueWriteBuffer(que, graph_edges_GPU, CL_TRUE,
		0, edges_memsize, DAG->adj,
		0, NULL, &write_edges_evt);
	ocl_check(err, "write dataset edges into graph_edges_GPU");
	err = clEnqueueWriteBuffer(que, n_entrypoints_GPU, CL_TRUE,
		0, sizeof(int), n_entrypoints,
		0, NULL, &write_nentries_evt);
	ocl_check(err, "write n_entries into n_entrypoints_GPU");
	err = clEnqueueWriteBuffer(que, entrypoints_GPU, CL_TRUE,
		0, entrypoints_memsize, entrypoints,
		0, NULL, &write_entries_evt);
	ocl_check(err, "write n_entries into entrypoints_GPU");

	//ESEGUIRE L'ALGORITMO DI SCHEDULING SU GPU
	cl_event entry_discover_evt = _entry_discover(n_nodes, graph_edges_GPU, n_entrypoints_GPU, entrypoints_GPU);

	//PASSA I RISULTATI DELLO SCHEDULING DALLA GPU ALLA CPU
	cl_event read_entries_evt;
	err = clEnqueueReadBuffer(que, entrypoints_GPU, CL_TRUE,
		0, entrypoints_memsize, entrypoints,
		1, &entry_discover_evt, &read_entries_evt);
	ocl_check(err, "read buffer entrypoints_GPU");
	cl_event read_nentries_evt;
	err = clEnqueueReadBuffer(que, n_entrypoints_GPU, CL_TRUE,
		0, sizeof(int), n_entrypoints,
		1, &entry_discover_evt, &read_nentries_evt);
	ocl_check(err, "read buffer n_entrypoints_GPU");

	//PULIZIA FINALE
	clReleaseKernel(entry_discover_k);
	return entry_discover_evt;
}

cl_event _compute_metrics(cl_mem graph_nodes_GPU, cl_mem queue_GPU, cl_mem next_queue_GPU, cl_mem graph_edges_GPU, cl_mem metrics_GPU)
{
	int arg_index = 0;
	int const n_nodes = DAG->len;
	const size_t lws[] = { preferred_wg_size };
	const size_t gws[] = { round_mul_up(n_nodes, lws[0]) };

	cl_int err;
	err = clSetKernelArg(compute_metrics_k, arg_index++, sizeof(graph_nodes_GPU), &graph_nodes_GPU);
	ocl_check(err,  "set arg %d for entry_discover_k", arg_index);
	err = clSetKernelArg(compute_metrics_k, arg_index++, sizeof(queue_GPU), &queue_GPU);
	ocl_check(err,  "set arg %d for entry_discover_k", arg_index);
	err = clSetKernelArg(compute_metrics_k, arg_index++, sizeof(next_queue_GPU), &next_queue_GPU);
	ocl_check(err,  "set arg %d for entry_discover_k", arg_index);
	err = clSetKernelArg(compute_metrics_k, arg_index++, sizeof(n_nodes), &n_nodes);
	ocl_check(err,  "set arg %d for entry_discover_k", arg_index);
	err = clSetKernelArg(compute_metrics_k, arg_index++, sizeof(graph_edges_GPU), &graph_edges_GPU);
	ocl_check(err,  "set arg %d for entry_discover_k", arg_index);
	err = clSetKernelArg(compute_metrics_k, arg_index++, sizeof(metrics_GPU), &metrics_GPU);
	ocl_check(err,  "set arg %d for entry_discover_k", arg_index);

	cl_event compute_metrics_evt;
	err = clEnqueueNDRangeKernel(que, compute_metrics_k,
		1, NULL, gws, lws,
		0, NULL, &compute_metrics_evt);

	ocl_check(err, "enqueue compute_metrics");

	return compute_metrics_evt;
}

int *queue_count , *next_queue_count, *visited;
cl_int4 *next_queue_4, *queue_4;
size_t queue_memsize, nodes_memsize, edges_memsize, visited_memsize, metrics_memsize;
cl_int2 *metrics, *ordered_metrics;

//CREO MEMORIA BUFFER IN GPU
cl_mem graph_nodes_GPU, queue_GPU, queue_count_GPU, next_queue_GPU, next_queue_count_GPU, metrics_GPU, ordered_metrics_GPU;

cl_event* compute_metrics_int4(){
	int const n_nodes = DAG->len;
	int const queue_len = ceil(n_nodes/4.0);
	const int metrics_len = m_N_padded; //necessario usare il round alla prossima potenza del due perché altrimenti il sort non potrebbe funzionare
	
	queue_4 = new cl_int4[queue_len];


	for (int i = 0; i < queue_len; i++)
	{
		int j = i*4;
		queue_4[i].x = (j<n_nodes)?entrypoints[j++]:0;
		queue_4[i].y = (j<n_nodes)?entrypoints[j++]:0;
		queue_4[i].z = (j<n_nodes)?entrypoints[j++]:0;
		queue_4[i].w = (j<n_nodes)?entrypoints[j]:0;
	}

	next_queue_4 = new cl_int4[queue_len]; for (int i = 0; i < queue_len; i++) next_queue_4[i] = cl_int4{0,0,0,0};
	queue_memsize = queue_len*sizeof(cl_int4);
	nodes_memsize = n_nodes*sizeof(int);
	edges_memsize = n_nodes*n_nodes*sizeof(bool);
	metrics_memsize = metrics_len*sizeof(cl_int2);

	//Creo e inizializzo memoria host e device per array che conterrà le metriche.
	//le metriche vengono inizializzate con il numero di parent in y per necessità del kernel.
	if(metrics_len < (DAG->len)) error("array metrics più piccolo di nodes array");
	metrics = new cl_int2[metrics_len]; 
	for (int i = 0; i < metrics_len; i++) { 
		metrics[i].x = 0; metrics[i].y = 0; 
		if(i > n_nodes) continue;
		for (int j = 0; j < n_nodes; j++) {
			int parent_of_node = DAG->adj[matrix_to_array_indexes(j, i, n_nodes)];
			if(parent_of_node > 0) metrics[i].y++;
		}
	}
	
	//CREO MEMORIA BUFFER IN GPU
	graph_nodes_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nodes_memsize, NULL, &err);
	ocl_check(err, "create buffer graph_nodes_GPU");
	queue_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, queue_memsize, NULL, &err);
	ocl_check(err, "create buffer queue_GPU");
	next_queue_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, queue_memsize, NULL, &err);
	ocl_check(err, "create buffer next_queue_GPU");
	metrics_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, metrics_memsize, NULL, &err);
	ocl_check(err, "create buffer metrics_GPU");
	//PASSARE I DATI DELLA DAG ALLA GPU 
	cl_event write_nodes_evt, write_queue_evt, write_next_queue_evt, write_edges_evt,  write_metrics_evt, write_visited_evt;
	err = clEnqueueWriteBuffer(que, graph_nodes_GPU, CL_TRUE,0, nodes_memsize, DAG->nodes,0, NULL, &write_nodes_evt);
	ocl_check(err, "write dataset nodes into graph_nodes_GPU");
	err = clEnqueueWriteBuffer(que, queue_GPU, CL_TRUE,0, queue_memsize, queue_4,0, NULL, &write_queue_evt);
	ocl_check(err, "write into queue_GPU");
	err = clEnqueueWriteBuffer(que, next_queue_GPU, CL_TRUE,0, queue_memsize, next_queue_4,0, NULL, &write_next_queue_evt);
	ocl_check(err, "write into next_queue_GPU");
	err = clEnqueueWriteBuffer(que, metrics_GPU, CL_TRUE,0, metrics_memsize, metrics,0, NULL, &write_metrics_evt);
	ocl_check(err, "write into metrics_GPU");
	//graph_edges_GPU già dichiarato al passo precedente, non duplico.

	//howManyGreater(queue_4, queue_len, 0);

	cl_event compute_metrics_evt_start, compute_metrics_evt_end;
	compute_metrics_evt_end = compute_metrics_evt_start = _compute_metrics(graph_nodes_GPU, queue_GPU, next_queue_GPU, graph_edges_GPU, metrics_GPU);
	//se non vogliamo farlo ad ogni ciclo, dovremmo mantenere un bool nel kernel che viene settato a true appena qualcuno scrive in next_queue, quindi invece di tutto next_queue possiamo leggere ad ogni ciclo solo il bit.
	cl_event read_next_queue_evt, read_metrics_evt;
	//printMemoryUsage();
	err = clEnqueueReadBuffer(que, next_queue_GPU, CL_TRUE, 0, queue_memsize, next_queue_4, 1, &compute_metrics_evt_start, &read_next_queue_evt);
	ocl_check(err, "read buffer next_queue_GPU 0");
	//printInt4(next_queue_4, queue_len, "\n", true);


	int count = 1;

	//TODO: questo continuo scambiare array giganti tra host e device rende inutile tutto il lavoro, lavorare su buffer condiviso, o spostare questa logica sul kernel.
	// credo sia possibile spostare questo loop nel kernel aggiungendo una barrier alla fine del ciclo.
	// potremmo anche avere un work item(quello con id 0 ad esempio) che si occupa di coordinare gli altri se serve.
	while(!isEmpty(next_queue_4, queue_len, cl_int4{0,0,0,0})) {
		//scambio le due code e reinizializzo la next_queue in modo da poter essere riutilizzata.
		//queue_GPU = next_queue_GPU;
		swap(queue_GPU, next_queue_GPU);
		swap(queue_4, next_queue_4);
		for (int i = 0; i < queue_len; i++) next_queue_4[i] = cl_int4{0,0,0,0};

		//passo il next_array reinizializzato alla GPU.
		//next_queue_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, queue_memsize, NULL, &err);
		//ocl_check(err, "create buffer next_queue_GPU");
		err = clEnqueueWriteBuffer(que, next_queue_GPU, CL_TRUE, 0, queue_memsize, next_queue_4 ,0, NULL, &write_next_queue_evt);
		ocl_check(err, "write into next_queue_GPU {0}", count);

		//O(n_nodes); => O(n_nodes^3)
		compute_metrics_evt_end = _compute_metrics(graph_nodes_GPU, queue_GPU, next_queue_GPU, graph_edges_GPU, metrics_GPU);

		//leggo il nuovo next_array dalla GPU.
		err = clEnqueueReadBuffer(que, next_queue_GPU, CL_TRUE, 0, queue_memsize, next_queue_4, 1, &compute_metrics_evt_end, &read_next_queue_evt);
		ocl_check(err, "read buffer next_queue_GPU {0}", count);

		//printInt4(next_queue_4, queue_len);

		count++;
	}

	cout<<"count:"<<count<<endl; //1278 vs 47

	//PASSA I RISULTATI DALLA GPU ALLA CPU
	err = clEnqueueReadBuffer(que, metrics_GPU, CL_TRUE,
		0, metrics_memsize, metrics,
		1, &compute_metrics_evt_end, &read_metrics_evt);
	ocl_check(err, "read buffer metrics_GPU");

	//PULIZIA FINALE
	clReleaseMemObject(queue_GPU);
	clReleaseMemObject(queue_count_GPU);
	clReleaseMemObject(next_queue_GPU);
	clReleaseMemObject(next_queue_count_GPU);
	clReleaseMemObject(graph_edges_GPU);

	clReleaseKernel(compute_metrics_k);

	// free(queue_4);
	// free(next_queue_4);

	cl_event *compute_metrics_evt = new cl_event[2];
	compute_metrics_evt[0] = compute_metrics_evt_start;
	compute_metrics_evt[1] = compute_metrics_evt_end;
	return compute_metrics_evt;
}

int *queue, *next_queue;
cl_event* compute_metrics(){
	int const n_nodes = DAG->len;
	const int metrics_len = m_N_padded; //necessario usare il round alla prossima potenza del due perché altrimenti il sort non potrebbe funzionare
	queue = entrypoints;
	next_queue = new int[n_nodes]; for (int i = 0; i < n_nodes; i++) next_queue[i] = 0;
	nodes_memsize = n_nodes*sizeof(int);
	queue_memsize = n_nodes*sizeof(int);
	edges_memsize = n_nodes*n_nodes*sizeof(bool);
	metrics_memsize = metrics_len*sizeof(cl_int2);

	//Creo e inizializzo memoria host e device per array che conterrà le metriche.
	//le metriche vengono inizializzate con il numero di parent in y per necessità del kernel.
	if(metrics_len < (DAG->len)) error("array metrics più piccolo di nodes array");
	metrics = new cl_int2[metrics_len]; 
	for (int i = 0; i < metrics_len; i++) { 
		metrics[i].x = 0; metrics[i].y = 0; 
		if(i > n_nodes) continue;
		for (int j = 0; j < n_nodes; j++) {
			int parent_of_node = DAG->adj[matrix_to_array_indexes(j, i, n_nodes)];
			if(parent_of_node > 0) metrics[i].y++;
		}
	}
	
	//CREO MEMORIA BUFFER IN GPU
	graph_nodes_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nodes_memsize, NULL, &err);
	ocl_check(err, "create buffer graph_nodes_GPU");
	queue_GPU = clCreateBuffer(ctx, CL_MEM_READ_ONLY, queue_memsize, NULL, &err);
	ocl_check(err, "create buffer queue_GPU");
	next_queue_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, queue_memsize, NULL, &err);
	ocl_check(err, "create buffer next_queue_GPU");
	metrics_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, metrics_memsize, NULL, &err);
	ocl_check(err, "create buffer metrics_GPU");
	//PASSARE I DATI DELLA DAG ALLA GPU 
	cl_event write_nodes_evt, write_queue_evt, write_next_queue_evt, write_edges_evt,  write_metrics_evt, write_visited_evt;
	err = clEnqueueWriteBuffer(que, graph_nodes_GPU, CL_TRUE,
		0, nodes_memsize, DAG->nodes,
		0, NULL, &write_nodes_evt);
	ocl_check(err, "write dataset nodes into graph_nodes_GPU");
	err = clEnqueueWriteBuffer(que, queue_GPU, CL_TRUE,
		0, queue_memsize, queue,
		0, NULL, &write_queue_evt);
	ocl_check(err, "write into queue_GPU");
	err = clEnqueueWriteBuffer(que, next_queue_GPU, CL_TRUE,
		0, queue_memsize, next_queue,
		0, NULL, &write_next_queue_evt);
	ocl_check(err, "write into next_queue_GPU");
	err = clEnqueueWriteBuffer(que, metrics_GPU, CL_TRUE,
		0, metrics_memsize, metrics,
		0, NULL, &write_metrics_evt);
	ocl_check(err, "write into metrics_GPU");
	err = clEnqueueWriteBuffer(que, metrics_GPU, CL_TRUE,
		0, metrics_memsize, metrics,
		0, NULL, &write_metrics_evt);
	ocl_check(err, "write into metrics_GPU");
	//graph_edges_GPU già dichiarato al passo precedente, non duplico.

	howManyGreater(queue, n_nodes, 0);

	cl_event compute_metrics_evt_start, compute_metrics_evt_end;
	compute_metrics_evt_end = compute_metrics_evt_start = _compute_metrics(graph_nodes_GPU, queue_GPU, next_queue_GPU, graph_edges_GPU, metrics_GPU);
	//se non vogliamo farlo ad ogni ciclo, dovremmo mantenere un bool nel kernel che viene settato a true appena qualcuno scrive in next_queue, quindi invece di tutto next_queue possiamo leggere ad ogni ciclo solo il bit.
	cl_event read_next_queue_evt, read_metrics_evt;
	err = clEnqueueReadBuffer(que, next_queue_GPU, CL_TRUE, 0, queue_memsize, next_queue, 1, &compute_metrics_evt_start, &read_next_queue_evt);
	ocl_check(err, "read buffer next_queue_GPU");
	

	int count = 1;
	// COMPUTE_METRICS_SECOND_IMPLEMENTATION
	// while(!isEmpty(next_queue, n_nodes, 0)) {
	// 	//scambio le due code e reinizializzo la next_queue in modo da poter essere riutilizzata.
	// 	queue_GPU = next_queue_GPU;
	// 	free(next_queue);
	// 	next_queue = new int[n_nodes]; for (int i = 0; i < n_nodes; i++) next_queue[i] = 0;

	// 	//passo il next_array reinizializzato alla GPU.
	// 	clReleaseMemObject(next_queue_GPU);
	// 	next_queue_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, queue_memsize, NULL, &err);
	// 	ocl_check(err, "create buffer next_queue_GPU");
	// 	err = clEnqueueWriteBuffer(que, next_queue_GPU, CL_TRUE, 0, queue_memsize, next_queue,0, NULL, &write_next_queue_evt);
	// 	ocl_check(err, "write into next_queue_GPU");

	// 	compute_metrics_evt_end = _compute_metrics(graph_nodes_GPU, queue_GPU, next_queue_GPU, graph_edges_GPU, metrics_GPU);

	// 	//leggo il nuovo next_array dalla GPU.
	// 	err = clEnqueueReadBuffer(que, next_queue_GPU, CL_TRUE, 0, queue_memsize, next_queue, 1, &compute_metrics_evt_end, &read_next_queue_evt);
	// 	ocl_check(err, "read buffer next_queue_GPU");
	// 	count++;
	// }

	// O(n_nodes^2) 
	while(!isEmpty(next_queue, n_nodes, 0)) {
		//scambio le due code e reinizializzo la next_queue in modo da poter essere riutilizzata.
		//queue_GPU = next_queue_GPU;
		swap(queue_GPU, next_queue_GPU);
		swap(queue, next_queue);
		for (int i = 0; i < n_nodes; i++) next_queue[i] = 0;

		//passo il next_array reinizializzato alla GPU.
		//next_queue_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, queue_memsize, NULL, &err);
		//ocl_check(err, "create buffer next_queue_GPU");
		err = clEnqueueWriteBuffer(que, next_queue_GPU, CL_TRUE, 0, queue_memsize, next_queue,0, NULL, &write_next_queue_evt);
		ocl_check(err, "write into next_queue_GPU");

		//O(n_nodes); => O(n_nodes^3)
		compute_metrics_evt_end = _compute_metrics(graph_nodes_GPU, queue_GPU, next_queue_GPU, graph_edges_GPU, metrics_GPU);

		//leggo il nuovo next_array dalla GPU.
		err = clEnqueueReadBuffer(que, next_queue_GPU, CL_TRUE, 0, queue_memsize, next_queue, 1, &compute_metrics_evt_end, &read_next_queue_evt);
		ocl_check(err, "read buffer next_queue_GPU");

		count++;
	}

	cout<<"count:"<<count<<endl; //1278 vs 47

	//PASSA I RISULTATI DALLA GPU ALLA CPU
	err = clEnqueueReadBuffer(que, metrics_GPU, CL_TRUE,
		0, metrics_memsize, metrics,
		1, &compute_metrics_evt_end, &read_metrics_evt);
	ocl_check(err, "read buffer metrics_GPU");

	//PULIZIA FINALE
	clReleaseMemObject(queue_GPU);
	clReleaseMemObject(queue_count_GPU);
	clReleaseMemObject(next_queue_GPU);
	clReleaseMemObject(next_queue_count_GPU);
	clReleaseMemObject(graph_edges_GPU);

	clReleaseKernel(compute_metrics_k);

	//free(entrypoints); //non necessario perché uno dei due fra queue e next_queue punta a entrypoints, facendo quindi il free di quei due viene liberato automaticamente anche questo.
	// free(queue);
	// free(next_queue);

	cl_event *compute_metrics_evt = new cl_event[2];
	compute_metrics_evt[0] = compute_metrics_evt_start;
	compute_metrics_evt[1] = compute_metrics_evt_end;
	return compute_metrics_evt;
}

int task_event_launched = 0;
#define MERGESORT_SMALL_STRIDE 1024 
cl_event* Sort_Mergesort()
{ 
	int c = 0;
	const int metrics_len = m_N_padded;
	cl_event temp;
	cl_event *sort_task_evts = new cl_event[32];
	task_event_launched = 0;
	cl_context Context = ctx; 
	cl_command_queue CommandQueue = que;
	size_t LocalWorkSize[3] = { preferred_wg_size,1,1 }; 
	size_t globalWorkSize[1];
	size_t localWorkSize[1];

	//Creo il buffer d'appoggio da passare alla GPU per il merge sort.
	cl_event write_ordered_metrics_evt;
	ordered_metrics_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, metrics_memsize, NULL, &err);
	ocl_check(err, "create buffer ordered_metrics_GPU");
	ordered_metrics = new cl_int2[metrics_len]; for (int i = 0; i < metrics_len; i++) ordered_metrics[i] = metrics[i];
	err = clEnqueueWriteBuffer(que, ordered_metrics_GPU, CL_TRUE,
		0, metrics_memsize, ordered_metrics,
		0, NULL, &write_ordered_metrics_evt);
	ocl_check(err, "write into metrics_GPU");

	//Comincio l'algoritmo di sorting.
	localWorkSize[0] = LocalWorkSize[0];
	globalWorkSize[0] = GetGlobalWorkSize(m_N_padded/2, localWorkSize[0]);
	unsigned int locLimit = 1;

	// if (m_N_padded >= LocalWorkSize[0] * 2) {
	// 	locLimit = 2 * LocalWorkSize[0];

	// 	// start with a local variant first, ASSUMING we have more than localWorkSize[0] * 2 elements
	// 	err = clSetKernelArg(m_MergesortStartKernel, 0, sizeof(cl_mem), (void*)&metrics_GPU);
	// 	err |= clSetKernelArg(m_MergesortStartKernel, 1, sizeof(cl_mem), (void*)&ordered_metrics_GPU);
	// 	ocl_check(err, "Failed to set kernel args: MergeSortStart");

	// 	err = clEnqueueNDRangeKernel(CommandQueue, m_MergesortStartKernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, &sort_task_evts[task_event_launched++]);
	// 	ocl_check(err, "Error executing MergeSortStart kernel!");

	// 	swap(metrics_GPU, ordered_metrics_GPU);
	// }

	// proceed with the global variant
	unsigned int stride = 2 * locLimit;
	if (m_N_padded <= MERGESORT_SMALL_STRIDE) {
		//non funziona per grandi numeri perché crea dei buffer locali e si rischia un cl out of resources.
		cout<<"small kernel " << (int)m_N_padded<<endl;
		// set not changing arguments
		err = clSetKernelArg(m_MergesortGlobalSmallKernel, 3, sizeof(cl_uint), (void*)&m_N_padded);
		ocl_check(err, "Failed to set kernel args: MergeSortGlobal");

		for (; stride <= m_N_padded; stride <<= 1 /*stride x 2*/) { // crea i branch per il merge sort.
			//calculate work sizes
			size_t neededWorkers = m_N_padded / stride;

			localWorkSize[0] = min(LocalWorkSize[0], neededWorkers);
			globalWorkSize[0] = GetGlobalWorkSize(neededWorkers, localWorkSize[0]);

			err = clSetKernelArg(m_MergesortGlobalSmallKernel, 0, sizeof(metrics_GPU), (void*)&metrics_GPU);
			err |= clSetKernelArg(m_MergesortGlobalSmallKernel, 1, sizeof(ordered_metrics_GPU), (void*)&ordered_metrics_GPU);
			err |= clSetKernelArg(m_MergesortGlobalSmallKernel, 2, sizeof(cl_uint), (void*)&stride);
			ocl_check(err, "Failed to set kernel args: m_MergesortGlobalSmallKernel");

			err = clEnqueueNDRangeKernel(CommandQueue, m_MergesortGlobalSmallKernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, &sort_task_evts[task_event_launched++]);
			ocl_check(err, "Error executing kernel m_MergesortGlobalSmallKernel! - %d, %d", localWorkSize[0], globalWorkSize[0]);

			swap(metrics_GPU, ordered_metrics_GPU);
		}
	}
	else {
		cout<<"big kernel " << m_N_padded<<endl;
		// set not changing arguments
		err = clSetKernelArg(m_MergesortGlobalBigKernel, 3, sizeof(cl_uint), (void*)&m_N_padded);
		ocl_check(err, "Failed to set kernel args: MergeSortGlobal");

		for (; stride <= m_N_padded; stride <<= 1) {
			//calculate work sizes
			size_t neededWorkers = m_N_padded / stride;

			localWorkSize[0] = min(LocalWorkSize[0], neededWorkers);
			globalWorkSize[0] = GetGlobalWorkSize(neededWorkers, localWorkSize[0]);

			err = clSetKernelArg(m_MergesortGlobalBigKernel, 0, sizeof(metrics_GPU), (void*)&metrics_GPU);
			err |= clSetKernelArg(m_MergesortGlobalBigKernel, 1, sizeof(ordered_metrics_GPU), (void*)&ordered_metrics_GPU);
			err |= clSetKernelArg(m_MergesortGlobalBigKernel, 2, sizeof(cl_uint), (void*)&stride);
			ocl_check(err, "Failed to set kernel args: m_MergesortGlobalBigKernel");

			err = clEnqueueNDRangeKernel(CommandQueue, m_MergesortGlobalBigKernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, &sort_task_evts[task_event_launched++]);
			ocl_check(err, "Error executing kernel m_MergesortGlobalBigKernel!");

			//TODO: check this if
			if (stride >= 1024 * 1024) ocl_check(clFinish(CommandQueue), "Failed finish CommandQueue at mergesort for bigger strides.");
			swap(metrics_GPU, ordered_metrics_GPU);
		}
	}

	cl_event read_sorted_metrics_evt;
	err = clEnqueueReadBuffer(que, metrics_GPU, CL_TRUE, //turns out this is the right array to read from because the other one is missing the last step of course...
		0, metrics_memsize, ordered_metrics,
		1, &sort_task_evts[task_event_launched-1], &read_sorted_metrics_evt);
	ocl_check(err, "read buffer ordered_metrics_GPU");

	clReleaseMemObject(metrics_GPU);
	clReleaseMemObject(ordered_metrics_GPU);
	clReleaseKernel(m_MergesortStartKernel);
	clReleaseKernel(m_MergesortGlobalSmallKernel);
	clReleaseKernel(m_MergesortGlobalBigKernel);

	return sort_task_evts;
}



  /*************************************************/
 //-----------------------------------------------//
//-----------------------------------------------//

std::chrono::system_clock::time_point start_time;
std::chrono::system_clock::time_point end_time;

bool isVector4Version = false;

int main(int argc, char *argv[])
{

	if (argc < 2) {
		error("syntax: graph_init datasetName [vector4 version (false)]");
	}else{
		if(argc > 2)isVector4Version = (strcmp(argv[2],"true") == 0);
	}

	start_time = std::chrono::system_clock::now();

	if(isVector4Version)ocl_init("./graph_init.ocl","entry_discover", "compute_metrics_4");
	else ocl_init("./graph_init.ocl","entry_discover", "compute_metrics");
	
	if(isVector4Version)
		cout<<"vector4 version"<<endl;
	else cout<<"standard version"<<endl;
	
	//LEGGERE IL DATASET E INIZIALIZZARE LA DAG
	DAG = initDagWithDataSet(argv[1]);
	m_N_padded = pow(2, ceil(log(DAG->len)/log(2))); //padded to the next power of 2

	//printf("DAG initialized\n");
	//DAG->Print();
	//cout<<"\n";
	//print(DAG->adj, DAG->len, DAG->len, ", ");
	
	cl_event entry_discover_evt = entry_discover();
	printf("entries discovered\n");
	//cout<<"entrypoints: "<<n_entries<<endl;
	//print(entrypoints, DAG->len, ", ", true);
	//cout<<"\n";

	const int metrics_len = m_N_padded;
	cl_event *compute_metrics_evt;
	if(isVector4Version) compute_metrics_evt = compute_metrics_int4();
	else compute_metrics_evt = compute_metrics();
	printf("metrics computed\n");

	//cout<<"metrics: "<<metrics_len<<endl;
	//print(metrics, DAG->len, "\n", true);
	//cout<<"\n";

	cl_event *sort_task_evts = Sort_Mergesort();	
	printf("array sorted\n");

	cout<<"sorted: "<<endl;
	//print(ordered_metrics, DAG->len, "\n", true);
	//print(ordered_metrics, metrics_len, "\n"); //per vedere anche i dati aggiunti per padding
	//cout<<"\n";
	
	end_time = std::chrono::system_clock::now();

	//METRICHE
	measurePerformance(entry_discover_evt, compute_metrics_evt, sort_task_evts, DAG->len);
	//VERIFICA DELLA CORRETTEZZA
	verify();

	//PULIZIA FINALE
	free(DAG);
	//free(entrypoints);
	// free(metrics);
	// free(ordered_metrics);

	//system("PAUSE");
}



  /*************************************************/
 //-----------------------------------------------//
//-----------------------------------------------//



void ocl_init(char* progName, char* kernelNameEntryDiscover, char *kernelNameComputeMetrics){
	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	ctx = create_context(p, d);
	que = create_queue(ctx, d);
	cl_program prog = create_program(progName, ctx, d);
	cl_program prog2 = create_program("./sort.ocl", ctx, d);
		
	entry_discover_k = clCreateKernel(prog, kernelNameEntryDiscover, &err);
	ocl_check(err, "create kernel %s", kernelNameEntryDiscover);
	compute_metrics_k = clCreateKernel(prog, kernelNameComputeMetrics, &err);
	ocl_check(err, "create kernel %s", kernelNameComputeMetrics);

	preferred_wg_size = get_preferred_work_group_size_multiple(compute_metrics_k, que);

	m_MergesortStartKernel = clCreateKernel(prog2, "Sort_MergesortStart", &err);
	ocl_check(err, "create kernel %s", "Sort_MergesortStart");
	m_MergesortGlobalSmallKernel = clCreateKernel(prog2, "Sort_MergesortGlobalSmall", &err);
	ocl_check(err, "create kernel %s", "Sort_MergesortGlobalSmall");
	m_MergesortGlobalBigKernel = clCreateKernel(prog2, "Sort_MergesortGlobalBig", &err);
	ocl_check(err, "create kernel %s", "Sort_MergesortGlobalBig");
}

bool is_file_empty(FILE *fp){
	fseek (fp, 0, SEEK_END);
	int size = ftell(fp);	
	rewind(fp);
	return size == 0;
}

std::string exec(const char* cmd) {
    char buffer[128];
    std::string result = "";
    FILE* pipe = _popen(cmd, "r");
    if (!pipe) throw std::runtime_error("popen() failed!");
    try {
        while (fgets(buffer, sizeof buffer, pipe) != NULL) {
            result += buffer;
        }
    } catch (...) {
        _pclose(pipe);
        throw;
    }
    _pclose(pipe);
    return result;
}

void measurePerformance(cl_event entry_discover_evt,cl_event *compute_metrics_evt, cl_event *sort_task_evts, int nels){
	double runtime_discover_ms = runtime_ms(entry_discover_evt);
	double runtime_metrics_ms = total_runtime_ms(compute_metrics_evt[0], compute_metrics_evt[1]);
	double runtime_sorts_ms =  total_runtime_ms(sort_task_evts[0], sort_task_evts[task_event_launched-1]);

	//TODO: check the math as algorithms changed
	printf("discover entries: runtime %.4gms, %.4g GE/s, %.4g GB/s\n",
		runtime_discover_ms, nels/runtime_discover_ms/1.0e6, preferred_wg_size/runtime_discover_ms/1.0e6);
	printf("compute metrics: runtime %.4gms, %.4g GE/s, %.4g GB/s\n",
		runtime_metrics_ms, nels/runtime_metrics_ms/1.0e6, preferred_wg_size/runtime_metrics_ms/1.0e6);
	printf("sort tasks: runtime %.4gms, %.4g GE/s, %.4g GB/s\n",
		runtime_sorts_ms, nels/runtime_sorts_ms/1.0e6, preferred_wg_size/runtime_sorts_ms/1.0e6);

	std::time_t end_time_t = std::chrono::system_clock::to_time_t(end_time);
	std::chrono::duration<double> elapsed_seconds = end_time-start_time;
	char *end_date_time = std::ctime(&end_time_t); end_date_time[strcspn(end_date_time , "\n")] = 0;

	double total_elapsed_time_GPU = total_runtime_ms(entry_discover_evt, sort_task_evts[task_event_launched-1]);
	int platform_id = 0;
	char *platform_name = getSelectedPlatformInfo(platform_id);
	int device_id = 0;
	char *device_name = getSelectedDeviceInfo(device_id);
	
	int gpu_temperature = -1;
	string gpu_temperature_string = exec("get_current_gpu_temperature.cmd");
	if(gpu_temperature_string.length() != 0){
		size_t start_sub = gpu_temperature_string.find(':')+1, end_sub = gpu_temperature_string.find_last_of('C')-1;
		gpu_temperature_string = gpu_temperature_string.substr(start_sub, end_sub-start_sub);
		gpu_temperature = atoi(gpu_temperature_string.c_str());
	}
	int cpu_temperature = -1;
	string cpu_temperature_string = exec("get_current_cpu_temperature.cmd");
	if(cpu_temperature_string.length() != 0){
		cpu_temperature = atoi(cpu_temperature_string.c_str());
	}
	//stampo i file in un .csv per poter analizzare i dati successivamente.
	FILE *fp;
	fp = fopen("execution_results.csv", "a");
	if (fp == NULL) {
		printf("Error opening file!\n");
		exit(1);
	}
	if(is_file_empty(fp))
		fprintf(fp, "DATA, TOTAL RUN SECONDS CPU, TOTAL RUN SECONDS GPU, PLATFORM, DEVICE, DISCOVER ENTRIES RUNTIME MS, COMPUTE METRICS RUNTIME MS, SORT RUNTIME MS, N TASKS, PREFERRED WORK GROUP SIZE, GPU TEMPERATURE, CPU TEMPERATURE\n");
	
	fprintf(fp,"%s, %.4g, %.4g, %s, %s, %.4g, %.4g, %.4g, %d, %d, %d, %d\n", 
	end_date_time, elapsed_seconds.count(), total_elapsed_time_GPU/1000, platform_name, device_name, runtime_discover_ms, runtime_metrics_ms, runtime_sorts_ms, nels, preferred_wg_size, gpu_temperature, cpu_temperature);

	printMemoryUsage();
	
	fflush(fp);
	fclose(fp);
}

void printMemoryUsage(){
	//virtual memory used by program
	PROCESS_MEMORY_COUNTERS_EX pmc;
	GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
	SIZE_T virtualMemUsedByMe = pmc.PrivateUsage; //in bytes

	//ram used by program
	SIZE_T physMemUsedByMe = pmc.WorkingSetSize; //in bytes
 
	printf("Memory usage: %dMB, %dMB \n", virtualMemUsedByMe / 1000 / 1000, physMemUsedByMe / 1000 / 1000);
}

void measurePerformance(cl_event evt, int nels, string event_name = "event"){
	double evt_ms = runtime_ms(evt);
	printf("evt %s: runtime %.4gms, %.4g GE/s, %.4g GB/s\n",
		event_name, evt_ms, nels/evt_ms/1.0e6, preferred_wg_size/evt_ms/1.0e6);
}

//Una l è minore di r se hanno lo stesso livello, e l ha peso maggiore o se l ha livello minore di r.
bool operator<(const cl_int2& l, const cl_int2& r)
{
    if(l.y == r.y){
		return l.x > r.x; //peso maggiore
	}
	return l.y < r.y; //o livello più basso
}
bool operator> (const cl_int2& l, const cl_int2& r){ return (r < l); }
bool operator<=(const cl_int2& l, const cl_int2& r){ return !(l > r); }
bool operator>=(const cl_int2& l, const cl_int2& r){ return !(l < r); }

void verify() {
	//TODO: calcolare le metriche su CPU e verificare che siano identiche a quelle calcolate su GPU;
	//scandire ordered_metrics e verificare che sia ordinati
	for (int i = 0; i < DAG->len - 1; ++i) {
		if(ordered_metrics[i] > ordered_metrics[i+1]){
			fprintf(stderr, "ordered_metrics[%d] = (%d, %d) > ordered_metrics[%d] = (%d, %d)\n", i, ordered_metrics[i].x, ordered_metrics[i].y, i+1, ordered_metrics[i+1].x, ordered_metrics[i+1].y);
			error("mismatch");
		}
	}
	printf("Everything sorted, verified\n");
}