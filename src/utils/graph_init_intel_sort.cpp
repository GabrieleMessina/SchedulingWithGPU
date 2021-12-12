#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include "dag.h"
#include "graphnode.h"
#include "utils/print_stuff.h"

#include <windows.h>
#include <tchar.h>
#include <assert.h>
#include <math.h>
#include <algorithm>
#include <iterator>
#include <vector>
#include <map>

#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"


typedef unsigned int uint;
typedef	uint QuicksortFlag;
#define RUN_CPU_SORTS
#define HOST 1
#include "Quicksort.h"
#define READ_ALIGNMENT  4096 // Intel recommended alignment
#define WRITE_ALIGNMENT 4096 // Intel recommended alignment

using namespace std;

void error(char const *str)
{
	fprintf(stderr, "%s\n", str);
	exit(1);
}

cl_context ctx;
cl_command_queue que;
cl_int err;
cl_kernel entry_discover_k, compute_metrics_k, sort_task_k, intel_quick_sort_k;
void ocl_init(char *progName, char *kernelNameEntryDiscover, char *kernelNameComputeMetrics);

Graph<int> *DAG;
Graph<int>* initDagWithDataSet();


void measurePerformance(cl_event init_evt, cl_event read_nodes_evt, cl_event read_edges_evt, int nels, size_t memsize);
void verify(int n_nodes, int const *nodes, int **edges);

int *entrypoints; 
cl_event _entry_discover(int n_nodes, cl_mem graph_nodes_GPU, cl_mem graph_edges_GPU, cl_mem n_entrypoints_GPU, cl_mem entrypoints_GPU)
{
	int arg_index = 0;
	size_t gws[] = { n_nodes };

	cl_int err;
	err = clSetKernelArg(entry_discover_k, arg_index++, sizeof(n_nodes), &n_nodes);
	ocl_check(err,  "set arg %d for entry_discover_k", arg_index);
	err = clSetKernelArg(entry_discover_k, arg_index++, sizeof(graph_nodes_GPU), &graph_nodes_GPU);
	ocl_check(err,  "set arg %d for entry_discover_k", arg_index);
	err = clSetKernelArg(entry_discover_k, arg_index++, sizeof(graph_edges_GPU), &graph_edges_GPU);
	ocl_check(err,  "set arg %d for entry_discover_k", arg_index);
	err = clSetKernelArg(entry_discover_k, arg_index++, sizeof(n_entrypoints_GPU), &n_entrypoints_GPU);
	ocl_check(err,  "set arg %d for entry_discover_k", arg_index);
	err = clSetKernelArg(entry_discover_k, arg_index++, sizeof(entrypoints_GPU), &entrypoints_GPU);
	ocl_check(err,  "set arg %d for entry_discover_k", arg_index);

	cl_event entry_discover_evt;
	err = clEnqueueNDRangeKernel(que, entry_discover_k,
		1, NULL, gws, NULL,
		0, NULL, &entry_discover_evt);

	ocl_check(err, "enqueue entry_discover");

	return entry_discover_evt;
}


int entry_discover(){
	int const n_nodes = DAG->len;
	size_t nodes_memsize = n_nodes*sizeof(int);
	size_t edges_memsize = n_nodes*n_nodes*sizeof(int);

	int *n_entrypoints = new int(0);
	entrypoints = new int[n_nodes]; for (int i = 0; i < n_nodes; i++) entrypoints[i] = -2;
	size_t entrypoints_memsize = n_nodes*sizeof(int);

	//CREO MEMORIA BUFFER IN GPU
	cl_mem graph_nodes_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nodes_memsize, NULL, &err);
	ocl_check(err, "create buffer graph_nodes_GPU");
	cl_mem graph_edges_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, edges_memsize, NULL, &err);
	ocl_check(err, "create buffer graph_edges_GPU");
	cl_mem n_entrypoints_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(n_entrypoints), NULL, &err);
	ocl_check(err, "create buffer n_entrypoints_GPU");
	cl_mem entrypoints_GPU = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, entrypoints_memsize, NULL, &err);
	ocl_check(err, "create buffer entrypoints_GPU");

	//PASSARE I DATI DELLA DAG ALLA GPU 
	cl_event write_nodes_evt, write_edges_evt, write_nentries_evt, write_entries_evt;
	err = clEnqueueWriteBuffer(que, graph_nodes_GPU, CL_TRUE,
		0, nodes_memsize, DAG->nodes,
		0, NULL, &write_nodes_evt);
	ocl_check(err, "write dataset nodes into graph_nodes_GPU");
	int * edges_GPU = matrix_to_array(DAG->adj, n_nodes, n_nodes);
	err = clEnqueueWriteBuffer(que, graph_edges_GPU, CL_TRUE,
		0, edges_memsize, edges_GPU,
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
	cl_event entry_discover_evt = _entry_discover(n_nodes, graph_nodes_GPU, graph_edges_GPU, n_entrypoints_GPU, entrypoints_GPU);

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
	clReleaseMemObject(graph_nodes_GPU);
	clReleaseMemObject(graph_edges_GPU);
	clReleaseMemObject(n_entrypoints_GPU);
	clReleaseMemObject(entrypoints_GPU); //TODO: questo potrebbe non essere rilasciato ma essere passato al prossimo step
	clReleaseKernel(entry_discover_k);
	return *n_entrypoints;
}

cl_event _compute_metrics(cl_mem graph_nodes_GPU, cl_mem queue_GPU, cl_mem queue_count_GPU, cl_mem next_queue_GPU, cl_mem next_queue_count_GPU, cl_mem graph_edges_GPU, cl_mem metrics_GPU, cl_mem visited_GPU)
{
	int arg_index = 0;
	int const n_nodes = DAG->len;
	size_t gws[] = { DAG->len };

	cl_int err;
	err = clSetKernelArg(compute_metrics_k, arg_index++, sizeof(graph_nodes_GPU), &graph_nodes_GPU);
	ocl_check(err,  "set arg %d for entry_discover_k", arg_index);
	err = clSetKernelArg(compute_metrics_k, arg_index++, sizeof(queue_GPU), &queue_GPU);
	ocl_check(err,  "set arg %d for entry_discover_k", arg_index);
	err = clSetKernelArg(compute_metrics_k, arg_index++, sizeof(queue_count_GPU), &queue_count_GPU);
	ocl_check(err,  "set arg %d for entry_discover_k", arg_index);
	err = clSetKernelArg(compute_metrics_k, arg_index++, sizeof(next_queue_GPU), &next_queue_GPU);
	ocl_check(err,  "set arg %d for entry_discover_k", arg_index);
	err = clSetKernelArg(compute_metrics_k, arg_index++, sizeof(next_queue_count_GPU), &next_queue_count_GPU);
	ocl_check(err,  "set arg %d for entry_discover_k", arg_index);
	err = clSetKernelArg(compute_metrics_k, arg_index++, sizeof(n_nodes), &n_nodes);
	ocl_check(err,  "set arg %d for entry_discover_k", arg_index);
	err = clSetKernelArg(compute_metrics_k, arg_index++, sizeof(graph_edges_GPU), &graph_edges_GPU);
	ocl_check(err,  "set arg %d for entry_discover_k", arg_index);
	err = clSetKernelArg(compute_metrics_k, arg_index++, sizeof(metrics_GPU), &metrics_GPU);
	ocl_check(err,  "set arg %d for entry_discover_k", arg_index);
	err = clSetKernelArg(compute_metrics_k, arg_index++, sizeof(visited_GPU), &visited_GPU);
	ocl_check(err,  "set arg %d for entry_discover_k", arg_index);

	cl_event compute_metrics_evt;
	err = clEnqueueNDRangeKernel(que, compute_metrics_k,
		1, NULL, gws, NULL,
		0, NULL, &compute_metrics_evt);

	ocl_check(err, "enqueue compute_metrics");

	return compute_metrics_evt;
}

int *queue_count , *next_queue_count, *queue, *next_queue, *visited;
size_t queue_memsize, nodes_memsize, edges_memsize, visited_memsize, metrics_memsize;
cl_int2 *metrics, *ordered_metrics;

//CREO MEMORIA BUFFER IN GPU
cl_mem graph_nodes_GPU, graph_edges_GPU, queue_GPU, queue_count_GPU, next_queue_GPU, next_queue_count_GPU, metrics_GPU, visited_GPU, ordered_metrics_GPU;

void compute_metrics(int n_entries){
	int const n_nodes = DAG->len;
	next_queue_count = new int(0);
	queue_count = new int(n_entries);
	queue = entrypoints;
	next_queue = new int[n_nodes]; for (int i = 0; i < n_nodes; i++) next_queue[i] = -1;
	nodes_memsize = n_nodes*sizeof(int);
	queue_memsize = n_nodes*sizeof(int);
	edges_memsize = n_nodes*n_nodes*sizeof(int);

	//CREO MEMORIA BUFFER IN GPU
	graph_nodes_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nodes_memsize, NULL, &err);
	ocl_check(err, "create buffer graph_nodes_GPU");
	queue_GPU = clCreateBuffer(ctx, CL_MEM_READ_ONLY, queue_memsize, NULL, &err);
	ocl_check(err, "create buffer queue_GPU");
	next_queue_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, queue_memsize, NULL, &err);
	ocl_check(err, "create buffer next_queue_GPU");
	queue_count_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(queue_count), NULL, &err);
	ocl_check(err, "create buffer queue_count_GPU");
	next_queue_count_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(next_queue_count), NULL, &err);
	ocl_check(err, "create buffer next_queue_count_GPU");
	graph_edges_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, edges_memsize, NULL, &err);
	ocl_check(err, "create buffer graph_edges_GPU");

	//PASSARE I DATI DELLA DAG ALLA GPU 
	cl_event write_nodes_evt;
	err = clEnqueueWriteBuffer(que, graph_nodes_GPU, CL_TRUE,
		0, nodes_memsize, DAG->nodes,
		0, NULL, &write_nodes_evt);
	ocl_check(err, "write dataset nodes into graph_nodes_GPU");
	cl_event write_queue_evt, write_next_queue_evt, write_queuecount_evt, write_next_queuecount_evt, write_edges_evt,  write_metrics_evt, write_visited_evt;
	err = clEnqueueWriteBuffer(que, queue_GPU, CL_TRUE,
		0, queue_memsize, queue,
		0, NULL, &write_queue_evt);
	ocl_check(err, "write into queue_GPU");
	err = clEnqueueWriteBuffer(que, next_queue_GPU, CL_TRUE,
		0, queue_memsize, next_queue,
		0, NULL, &write_next_queue_evt);
	ocl_check(err, "write into next_queue_GPU");
	err = clEnqueueWriteBuffer(que, queue_count_GPU, CL_TRUE,
		0, sizeof(queue_count), queue_count,
		0, NULL, &write_queuecount_evt);
	ocl_check(err, "write into queue_count_GPU");
	err = clEnqueueWriteBuffer(que, next_queue_count_GPU, CL_TRUE,
		0, sizeof(next_queue_count), next_queue_count,
		0, NULL, &write_next_queuecount_evt);
	ocl_check(err, "write into next_queue_count_GPU");
	//TODO: duplicato, buffer già creato nel passo precedente dello scheduling
	int * edges_array = matrix_to_array(DAG->adj, n_nodes, n_nodes);
	err = clEnqueueWriteBuffer(que, graph_edges_GPU, CL_TRUE,
		0, edges_memsize, edges_array,
		0, NULL, &write_edges_evt);
	ocl_check(err, "write dataset edges into graph_edges_GPU");
	err = clEnqueueWriteBuffer(que, metrics_GPU, CL_TRUE,
		0, metrics_memsize, metrics,
		0, NULL, &write_metrics_evt);
	ocl_check(err, "write into metrics_GPU");

	//ESEGUIRE L'ALGORITMO DI SCHEDULING SU GPU
	cl_event compute_metrics_evt = _compute_metrics(graph_nodes_GPU, queue_GPU, queue_count_GPU, next_queue_GPU, next_queue_count_GPU, graph_edges_GPU, metrics_GPU, visited_GPU);

	//PASSA I RISULTATI DELLO SCHEDULING DALLA GPU ALLA CPU
	cl_event read_next_queue_evt, read_next_queue_count_evt, read_visited_evt, read_metrics_evt;
	err = clEnqueueReadBuffer(que, next_queue_GPU, CL_TRUE,
		0, queue_memsize, next_queue,
		1, &compute_metrics_evt, &read_next_queue_evt);
	ocl_check(err, "read buffer next_queue_GPU");
	err = clEnqueueReadBuffer(que, next_queue_count_GPU, CL_TRUE,
		0, sizeof(next_queue_count), next_queue_count,
		1, &compute_metrics_evt, &read_next_queue_count_evt);
	ocl_check(err, "read buffer next_queue_count_GPU");
	err = clEnqueueReadBuffer(que, visited_GPU, CL_TRUE,
		0, visited_memsize, visited,
		1, &compute_metrics_evt, &read_visited_evt);
	ocl_check(err, "read buffer visited_GPU");
	err = clEnqueueReadBuffer(que, metrics_GPU, CL_TRUE,
		0, metrics_memsize, metrics,
		1, &compute_metrics_evt, &read_metrics_evt);
	ocl_check(err, "read buffer metrics_GPU");

	if(*next_queue_count > 0) {
		entrypoints = next_queue;
		compute_metrics(*next_queue_count);
	}
}

cl_event _sort_task(cl_mem metrics_GPU, cl_mem sorted_metrics_GPU)
{
	int arg_index = 0;
	int const n_nodes = DAG->len;
	size_t gws[] = { DAG->len };
	//size_t gws[] = { 64 };

	cl_int err;
	err = clSetKernelArg(sort_task_k, arg_index++, sizeof(metrics_GPU), &metrics_GPU);
	ocl_check(err,  "set arg %d for sort_task_k", arg_index);
	err = clSetKernelArg(sort_task_k, arg_index++, sizeof(sorted_metrics_GPU), &sorted_metrics_GPU);
	ocl_check(err,  "set arg %d for sort_task_k", arg_index);

	cl_event sort_task_evt;
	err = clEnqueueNDRangeKernel(que, sort_task_k,
		1, NULL, gws, NULL,
		0, NULL, &sort_task_evt);

	ocl_check(err, "enqueue sort_task kernel");

	return sort_task_evt;
}

void sort_task(){
	int const n_nodes = DAG->len;
	ordered_metrics = new cl_int2[DAG->n];
	metrics_memsize = DAG->n*sizeof(cl_int2);

	cl_mem ordered_metrics_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, metrics_memsize, NULL, &err);
	ocl_check(err, "create buffer ordered_metrics_GPU");

	cl_event write_nodes_evt;
	err = clEnqueueWriteBuffer(que, ordered_metrics_GPU, CL_TRUE,
		0, metrics_memsize, ordered_metrics,
		0, NULL, &write_nodes_evt);
	ocl_check(err, "write into ordered_metrics_GPU");

	cl_event sort_task = _sort_task(metrics_GPU, ordered_metrics_GPU);

	cl_event read_sorted_metrics_evt;
	err = clEnqueueReadBuffer(que, ordered_metrics_GPU, CL_TRUE,
		0, metrics_memsize, ordered_metrics,
		1, &sort_task, &read_sorted_metrics_evt);
	ocl_check(err, "read buffer next_queue_GPU");
}

cl_int ciErrNum;
cl_mem db;
cl_mem dnb;
cl_mem blocksb;
cl_mem parentsb;
cl_mem newsb;
cl_mem workb;
cl_mem doneb;
template <class T>
void InstantiateOpenCLKernels(size_t size, const size_t MAXSEQ, const size_t MAX_SIZE, work_record* pdone, work_record* pnews, T* d)
{	
	cl_platform_id platform = select_platform();
	cl_device_id device = select_device(platform);
	cl_program prog = create_program("./QuicksortKernels.cl", ctx, device);
	intel_quick_sort_k = clCreateKernel(prog, "relauncher_kernel", &ciErrNum);
	CheckCLError (ciErrNum, "Kernel relauncher_kernel creation failed.", "Kernel created.");

	// Create buffer objects for memory.
	// allocate buffers
	db = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, ((sizeof(T)*size)/64 + 1)*64, d, &ciErrNum);
	CheckCLError (ciErrNum, "clCreateBuffer failed.", "clCreateBuffer.");
	dnb = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(T)*size, 0, &ciErrNum);
	CheckCLError (ciErrNum, "clCreateBuffer failed.", "clCreateBuffer.");
	blocksb = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(block_record)*MAX_SIZE, 0, &ciErrNum);
	CheckCLError (ciErrNum, "clCreateBuffer failed.", "clCreateBuffer.");
	parentsb = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(parent_record)*MAX_SIZE, 0, &ciErrNum);
	CheckCLError (ciErrNum, "clCreateBuffer failed.", "clCreateBuffer.");
	//cl_mem newsb = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(work_record)*MAX_SIZE, &news[0], &ciErrNum);
	newsb = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, ((sizeof(work_record)*MAX_SIZE)/64+1)*64, pnews, &ciErrNum);
	CheckCLError (ciErrNum, "clCreateBuffer failed.", "clCreateBuffer.");
	workb = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(work_record)*MAX_SIZE, 0, &ciErrNum);
	CheckCLError (ciErrNum, "clCreateBuffer failed.", "clCreateBuffer.");
	//cl_mem doneb = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(work_record)*MAX_SIZE, &done[0], &ciErrNum);
	doneb = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, ((sizeof(work_record)*MAX_SIZE)/64+1)*64, pdone, &ciErrNum);
	CheckCLError (ciErrNum, "clCreateBuffer failed.", "clCreateBuffer.");

	ciErrNum |= clSetKernelArg(intel_quick_sort_k, 0, sizeof(cl_mem), (void*) db);
	CheckCLError(ciErrNum, "clSetKernelArg failed.", "clSetKernelArg");
	ciErrNum |= clSetKernelArg(intel_quick_sort_k, 1, sizeof(cl_mem), (void*) dnb);
	CheckCLError(ciErrNum, "clSetKernelArg failed.", "clSetKernelArg");
	ciErrNum |= clSetKernelArg(intel_quick_sort_k, 2, sizeof(cl_mem), (void*) blocksb);
	CheckCLError(ciErrNum, "clSetKernelArg failed.", "clSetKernelArg");
	ciErrNum |= clSetKernelArg(intel_quick_sort_k, 3, sizeof(cl_mem), (void*) parentsb);
	CheckCLError(ciErrNum, "clSetKernelArg failed.", "clSetKernelArg");
	ciErrNum |= clSetKernelArg(intel_quick_sort_k, 4, sizeof(cl_mem), (void*) newsb);
	CheckCLError(ciErrNum, "clSetKernelArg failed.", "clSetKernelArg");
	ciErrNum |= clSetKernelArg(intel_quick_sort_k, 5, sizeof(cl_mem), (void*) workb);
	CheckCLError(ciErrNum, "clSetKernelArg failed.", "clSetKernelArg");
	ciErrNum |= clSetKernelArg(intel_quick_sort_k, 6, sizeof(cl_mem), (void*) doneb);
	CheckCLError(ciErrNum, "clSetKernelArg failed.", "clSetKernelArg");

	ciErrNum |= clSetKernelArg(intel_quick_sort_k, 8, sizeof(uint), (void*) &MAXSEQ);
	CheckCLError(ciErrNum, "clSetKernelArg failed.", "clSetKernelArg");
}

template <class T>
void GPUQSort(size_t size, T* d, work_record* pdone, work_record* pnews, const size_t MAX_SIZE)  {
	uint startpivot = median(d[0], d[size/2], d[size-1]);	
	
	for(uint i = 0; i < MAX_SIZE; i++) {
		new (pdone+i) work_record();
		if (i == 0)
			new (pnews) work_record(0, size, startpivot, 1);
		else
			new (pnews+i) work_record();
	}

	uint done_size = 0;
	ciErrNum |= clSetKernelArg(intel_quick_sort_k, 7, sizeof(uint), (void*) &done_size);
	CheckCLError(ciErrNum, "clSetKernelArg failed.", "clSetKernelArg");

	uint num_workgroups = 1;
	ciErrNum |= clSetKernelArg(intel_quick_sort_k, 9, sizeof(uint), (void*) &num_workgroups);
	CheckCLError(ciErrNum, "clSetKernelArg failed.", "clSetKernelArg");

	size_t		dimNDR = 1;	
	size_t gws[] = { DAG->len };
	ciErrNum = clEnqueueNDRangeKernel (que, intel_quick_sort_k, 1, NULL, gws, 0, 0, NULL, 0);
	CheckCLError(ciErrNum, "clEnqueueNDRangeKernel failed.", "clEnqueueNDRangeKernel");

	T* foo = (T*)clEnqueueMapBuffer(que, db, CL_TRUE, CL_MAP_READ, 0, sizeof(T)*size, 0, 0, 0, &ciErrNum); 
	CheckCLError(ciErrNum, "clEnqueueMapBuffer failed.", "clEnqueueMapBuffer");

	ciErrNum = clEnqueueUnmapMemObject(que, db, foo, 0, 0, 0);
	CheckCLError(ciErrNum, "clEnqueueUnmapMemObject failed.", "clEnqueueUnmapMemObject");
}

size_t optp(size_t s, double k, size_t m) {
	return (size_t)pow(2, floor(log(s*k + m)/log(2.0) + 0.5));
}

int main(int argc, char *argv[])
{

	if (argc != 2) {
		error("syntax: graph_init numberOfProcessors");
	}
	int n_processors = atoi(argv[1]);
	if (n_processors < 1)
		error("numberOfProcessors < 1");

	ocl_init("./graph_init.ocl","entry_discover", "compute_metrics");

	//LEGGERE IL DATASET E INIZIALIZZARE LA DAG
	DAG = initDagWithDataSet();
	DAG->Print();
	cout<<"\n";
	print(DAG->adj, DAG->n, DAG->n);
	
	int n_entries = entry_discover();
	cout<<"entrypoints: "<<endl;
	print(entrypoints, n_entries);
	cout<<"\n";

	metrics = new cl_int2[DAG->n];
	metrics_memsize = DAG->n*sizeof(cl_int2);
	visited = new int[DAG->n]; for (int i = 0; i < DAG->n; i++) visited[i] = 0;
	visited_memsize = DAG->n*sizeof(int);
	visited_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, visited_memsize, NULL, &err);
	ocl_check(err, "create buffer visited_GPU");
	metrics_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, metrics_memsize, NULL, &err);
	ocl_check(err, "create buffer metrics_GPU");
	cl_event write_visited_evt, write_metrics_evt;
	err = clEnqueueWriteBuffer(que, visited_GPU, CL_TRUE,
		0, visited_memsize, visited,
		0, NULL, &write_visited_evt);
	ocl_check(err, "write into visited_GPU");
	err = clEnqueueWriteBuffer(que, metrics_GPU, CL_TRUE,
		0, metrics_memsize, metrics,
		0, NULL, &write_metrics_evt);
	ocl_check(err, "write into metrics_GPU");
	compute_metrics(n_entries);
	cout<<"metrics: "<<endl;
	print(metrics, DAG->n);
	cout<<"\n";

	sort_task();	
	cout<<"sorted: "<<endl;
	print(ordered_metrics, DAG->n);
	cout<<"\n";

	// const size_t MAXSEQ = optp(DAG->n, 0.00009516, 203);
	// const size_t MAX_SIZE = 12*max(MAXSEQ, QUICKSORT_BLOCK_SIZE);
	// work_record* pdone = (work_record*)_aligned_malloc (((MAX_SIZE*sizeof(work_record))/64 + 1)*64, 4096);
	// work_record* pnews = (work_record*)_aligned_malloc (((MAX_SIZE*sizeof(work_record))/64 + 1)*64, 4096);
	// InstantiateOpenCLKernels (DAG->n, MAXSEQ, MAX_SIZE, pdone, pnews, metrics);
	// GPUQSort(DAG->n, metrics, pdone, pnews, MAX_SIZE);

	// cout<<"sorted: "<<endl;
	// print(metrics, DAG->n);
	// cout<<"\n";

	clReleaseMemObject(queue_GPU);
	clReleaseMemObject(queue_count_GPU);
	clReleaseMemObject(next_queue_GPU);
	clReleaseMemObject(next_queue_count_GPU);
	clReleaseMemObject(graph_edges_GPU);
	clReleaseMemObject(metrics_GPU);
	clReleaseMemObject(visited_GPU);
	clReleaseKernel(compute_metrics_k);
	

	clReleaseMemObject(db);
	clReleaseMemObject(dnb);
	clReleaseMemObject(blocksb);
	clReleaseMemObject(parentsb);
	clReleaseMemObject(newsb);
	clReleaseMemObject(workb);
	clReleaseMemObject(doneb);
	clReleaseKernel(intel_quick_sort_k);

	//ESEGUIRE L'ALGORITMO DI SCHEDULING SU GPU
	// cl_event entry_discover_evt = entry_discover(n_nodes, graph_nodes_GPU, graph_edges_GPU, n_entrypoints_GPU, entrypoints_GPU);
	/*passo entry a BFS come coda dei nodi da attraversare.
	cl_event entry_discover_evt = BFS(que, entry_discover_k, n_nodes, n_processors, graph_nodes_GPU, graph_edges_GPU);
	cl_event entry_discover_evt = schedule_on_processor(que, entry_discover_k, n_nodes, n_processors, graph_nodes_GPU, graph_edges_GPU);
	*/
	
	//METRICHE
	//measurePerformance(entry_discover_evt, read_nodes_evt,read_edges_evt, n_nodes, nodes_memsize);
	//VERIFICA DELLA CORRETTEZZA
	//verify(n_nodes, DAG->nodes, DAG->adj);

	//PULIZIA FINALE
	free(DAG);
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
	//cl_program prog2 = create_program("./QuicksortKernels.cl", ctx, d);
		
	entry_discover_k = clCreateKernel(prog, kernelNameEntryDiscover, &err);
	ocl_check(err, "create kernel %s", kernelNameEntryDiscover);
	compute_metrics_k = clCreateKernel(prog, kernelNameComputeMetrics, &err);
	ocl_check(err, "create kernel %s", kernelNameComputeMetrics);
	sort_task_k = clCreateKernel(prog2, "Sort_MergesortStart", &err);
	//sort_task_k = clCreateKernel(prog2, "relauncher_kernel", &err);
	ocl_check(err, "create kernel %s", "Sort_MergesortStart");

}


Graph<int>* initDagWithDataSet(){

	ifstream data_set;
	data_set.open("./data_set/first.txt");

	if(!data_set.is_open()){
	    error("impossibile aprire il dataset");
	}

	//int n_nodes = count(std::istreambuf_iterator<char>(data_set), istreambuf_iterator<char>(), '\n');
	int n_nodes = 0;
	{
		string temp;
		while (getline(data_set, temp))
		{
			if(temp.compare("-1") != 0 && temp.compare("") != 0)
				n_nodes++;
		}
	}
	data_set.clear();
	data_set.seekg(0);
		
	Graph<int> *DAG = new Graph<int>(n_nodes);
	if (!DAG) {
		data_set.close();
		error("failed to allocate graph");
	}

	//leggo tutti gli id in prima posizione in modo da creare la dag senza adj per il momento.
	int value, n_successor, successor, data_transfer;
	while(data_set >> value >> n_successor){
		int index = DAG->insertNode(value);
		for (int i = 0; i < n_successor; i++)
		{
			data_set>>successor>>data_transfer;
		}
	}

	data_set.clear();
	data_set.seekg(0);

	//adesso leggo di nuovo per creare la adj
	//TODO: e se mantenessi in memoria una matrice quadrata con tutti questi dati in fila per ogni task? IN modo da avere coalescenza?
	while(data_set >> value >> n_successor){
		int index = DAG->indexOfNode(value);
		for (int i = 0; i < n_successor; i++)
		{
			data_set>>successor>>data_transfer;
			DAG->insertEdgeByIndex(index, DAG->indexOfNode(successor), data_transfer); //TODO: al momento assumo che l'indice sia l'id dell'elemento, altrimenti avrei dovuto leggere i dati da dataset a partire dal fondo a causa delle dipendenze.
		}
	}

	data_set.close();
	return DAG;	
}


void measurePerformance(cl_event init_evt, cl_event read_nodes_evt, cl_event read_edges_evt, int nels, size_t memsize){
	double runtime_init_ms = runtime_ms(init_evt);
	double runtime_read_nodes_ms = runtime_ms(read_nodes_evt);
	double runtime_read_edges_ms = runtime_ms(read_edges_evt);

	//TODO: check the math as algorithms changed
	printf("init: runtime %.4gms, %.4g GE/s, %.4g GB/s\n",
		runtime_init_ms, nels/runtime_init_ms/1.0e6, memsize/runtime_init_ms/1.0e6);
	printf("read nodes: runtime %.4gms, %.4g GE/s, %.4g GB/s\n",
		runtime_read_nodes_ms, nels/runtime_read_nodes_ms/1.0e6, memsize/runtime_read_nodes_ms/1.0e6);
	printf("read edges: runtime %.4gms, %.4g GE/s, %.4g GB/s\n",
		runtime_read_edges_ms, nels/runtime_read_edges_ms/1.0e6, memsize/runtime_read_edges_ms/1.0e6);
}


void verify(int n_nodes, int const *nodes, int **edges) {
	if(nodes == NULL) error("nodes is null in verify");
	for (int i = 0; i < n_nodes; ++i) {
		if (nodes[i] != i) {
			fprintf(stderr, "%d != %d in nodes\n", i, nodes[i]);
			error("mismatch");
		}
	}
	if(edges == NULL || *edges == NULL) error("edges is null in verify");
	for (int i = 0; i < n_nodes; ++i) {
		if (edges[i][i] != i) {
			fprintf(stderr, "%d != %d in edges\n", i, edges[i][i]);
			error("mismatch");
		}
	}
}


