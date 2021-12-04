#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include "dag.h"
#include "graphnode.h"
#include "utils/print_stuff.h"

#include <math.h>
#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"

using namespace std;

void error(char const *str)
{
	fprintf(stderr, "%s\n", str);
	exit(1);
}

cl_context ctx;
cl_command_queue que;
cl_int err;
cl_kernel entry_discover_k, compute_metrics_k, sort_task_k, m_MergesortGlobalBigKernel, m_MergesortGlobalSmallKernel, m_MergesortStartKernel;
void ocl_init(char *progName, char *kernelNameEntryDiscover, char *kernelNameComputeMetrics);
size_t preferred_wg_size;

Graph<int> *DAG;
Graph<int>* initDagWithDataSet(string fileName);

void measurePerformance(cl_event init_evt, cl_event read_nodes_evt, cl_event read_edges_evt, int nels, size_t memsize);
void verify(int n_nodes, int const *nodes, int **edges);

int *entrypoints; 
cl_event _entry_discover(int n_nodes, cl_mem graph_edges_GPU, cl_mem n_entrypoints_GPU, cl_mem entrypoints_GPU)
{
	int arg_index = 0;
	size_t gws[] = { n_nodes }; //TODO: round mul up

	cl_int err;
	err = clSetKernelArg(entry_discover_k, arg_index++, sizeof(n_nodes), &n_nodes);
	ocl_check(err,  "set arg %d for entry_discover_k", arg_index);
	// err = clSetKernelArg(entry_discover_k, arg_index++, sizeof(graph_nodes_GPU), &graph_nodes_GPU);
	// ocl_check(err,  "set arg %d for entry_discover_k", arg_index);
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

cl_mem graph_edges_GPU;
cl_mem n_entrypoints_GPU;
cl_mem entrypoints_GPU;


int entry_discover(){
	int const n_nodes = DAG->len;
	size_t nodes_memsize = n_nodes*sizeof(int);
	size_t edges_memsize = n_nodes*n_nodes*sizeof(bool);

	int *n_entrypoints = new int(0);
	entrypoints = new int[n_nodes]; for (int i = 0; i < n_nodes; i++) entrypoints[i] = -1;
	size_t entrypoints_memsize = n_nodes*sizeof(int);

	//CREO MEMORIA BUFFER IN GPU
	// cl_mem graph_nodes_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nodes_memsize, NULL, &err);
	// ocl_check(err, "create buffer graph_nodes_GPU");
	graph_edges_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, edges_memsize, NULL, &err);
	ocl_check(err, "create buffer graph_edges_GPU");
	n_entrypoints_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(n_entrypoints), NULL, &err);
	ocl_check(err, "create buffer n_entrypoints_GPU");
	entrypoints_GPU = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, entrypoints_memsize, NULL, &err);
	ocl_check(err, "create buffer entrypoints_GPU");

	//PASSARE I DATI DELLA DAG ALLA GPU 
	cl_event write_nodes_evt, write_edges_evt, write_nentries_evt, write_entries_evt;
	// err = clEnqueueWriteBuffer(que, graph_nodes_GPU, CL_TRUE,
	// 	0, nodes_memsize, DAG->nodes,
	// 	0, NULL, &write_nodes_evt);
	// ocl_check(err, "write dataset nodes into graph_nodes_GPU");
	bool * edges_GPU = matrix_to_array(DAG->adj, n_nodes, n_nodes);
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
	// clReleaseMemObject(graph_nodes_GPU);
	// clReleaseMemObject(graph_edges_GPU);
	// clReleaseMemObject(n_entrypoints_GPU);
	// clReleaseMemObject(entrypoints_GPU); //TODO: questo potrebbe non essere rilasciato ma essere passato al prossimo step
	clReleaseKernel(entry_discover_k);
	return *n_entrypoints;
}

cl_event _compute_metrics(cl_mem graph_nodes_GPU, cl_mem queue_GPU, cl_mem queue_count_GPU, cl_mem next_queue_GPU, cl_mem next_queue_count_GPU, cl_mem graph_edges_GPU, cl_mem metrics_GPU)
{
	int arg_index = 0;
	int const n_nodes = DAG->len;
	//size_t lws[] = { n_nodes * 2 }; //number of work item to launch in the same work group, that means they are garanted to be executed at the same time.
	//size_t gws[] = {GetGlobalWorkSize(n_nodes, lws[0])}; //total work item to launch.
	const size_t gws[] = { n_nodes }; //TODO: round mul up
	//const size_t gws[] = { round_mul_up(n_nodes, preferred_wg_size) };

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
cl_mem graph_nodes_GPU, queue_GPU, queue_count_GPU, next_queue_GPU, next_queue_count_GPU, metrics_GPU, ordered_metrics_GPU;

void compute_metrics(int n_entries){
	int const n_nodes = DAG->len;
	next_queue_count = new int(0);
	queue_count = new int(n_entries);
	queue = entrypoints;
	next_queue = new int[n_nodes]; for (int i = 0; i < n_nodes; i++) next_queue[i] = -1;
	nodes_memsize = n_nodes*sizeof(int);
	queue_memsize = n_nodes*sizeof(int);
	edges_memsize = n_nodes*n_nodes*sizeof(bool);

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
	// graph_edges_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, edges_memsize, NULL, &err);
	// ocl_check(err, "create buffer graph_edges_GPU");

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
	// bool * edges_array = matrix_to_array(DAG->adj, n_nodes, n_nodes);
	// err = clEnqueueWriteBuffer(que, graph_edges_GPU, CL_TRUE,
	// 	0, edges_memsize, edges_array,
	// 	0, NULL, &write_edges_evt);
	// ocl_check(err, "write dataset edges into graph_edges_GPU");
	err = clEnqueueWriteBuffer(que, metrics_GPU, CL_TRUE,
		0, metrics_memsize, metrics,
		0, NULL, &write_metrics_evt);
	ocl_check(err, "write into metrics_GPU");

	//ESEGUIRE L'ALGORITMO DI SCHEDULING SU GPU
	cl_event compute_metrics_evt = _compute_metrics(graph_nodes_GPU, queue_GPU, queue_count_GPU, next_queue_GPU, next_queue_count_GPU, graph_edges_GPU, metrics_GPU);

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
	// err = clEnqueueReadBuffer(que, visited_GPU, CL_TRUE,
	// 	0, visited_memsize, visited,
	// 	1, &compute_metrics_evt, &read_visited_evt);
	// ocl_check(err, "read buffer visited_GPU");
	err = clEnqueueReadBuffer(que, metrics_GPU, CL_TRUE,
		0, metrics_memsize, metrics,
		1, &compute_metrics_evt, &read_metrics_evt);
	ocl_check(err, "read buffer metrics_GPU");

	// if(*next_queue_count > 0) {
	// 	entrypoints = next_queue;
	// 	compute_metrics(*next_queue_count);
	// }
	
	if(!isEmpty(next_queue, n_nodes, -1)) {
		entrypoints = next_queue;
		compute_metrics(*next_queue_count);
	}
}

#define MERGESORT_SMALL_STRIDE 1024 * 64
size_t m_N_padded;
void Sort_Mergesort()
{ 
	int c = 0;
	cl_event sort_task_evt;
	cl_context Context = ctx; 
	cl_command_queue CommandQueue = que;
	size_t LocalWorkSize[3] = { m_N_padded,1,1 }; //TODO:il lws necessita di essere un multiplo di m_N_padded, suppongo a causa dell'implementazione, è un problema?
	//TODO fix memory problem when many elements. -> CL_OUT_OF_RESOURCES
	size_t globalWorkSize[1];
	size_t localWorkSize[1];

	localWorkSize[0] = LocalWorkSize[0];
	globalWorkSize[0] = GetGlobalWorkSize(m_N_padded/2, localWorkSize[0]);
	unsigned int locLimit = 1;

	if (m_N_padded >= LocalWorkSize[0] * 2) {
		cout<<"MergesortStartKernel " << m_N_padded<<endl;
		locLimit = 2 * LocalWorkSize[0];

		// start with a local variant first, ASSUMING we have more than localWorkSize[0] * 2 elements
		err = clSetKernelArg(m_MergesortStartKernel, 0, sizeof(metrics_GPU), (void*)&metrics_GPU);
		err |= clSetKernelArg(m_MergesortStartKernel, 1, sizeof(ordered_metrics_GPU), (void*)&ordered_metrics_GPU);
		ocl_check(err, "Failed to set kernel args: MergeSortStart");

		err = clEnqueueNDRangeKernel(CommandQueue, m_MergesortStartKernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, &sort_task_evt);
		ocl_check(err, "Error executing MergeSortStart kernel!");

		swap(metrics, ordered_metrics);
	}

	// proceed with the global variant
	unsigned int stride = 2 * locLimit;

	if (m_N_padded <= MERGESORT_SMALL_STRIDE) {
		cout<<"small kernel " << m_N_padded<<endl;
		// set not changing arguments
		err = clSetKernelArg(m_MergesortGlobalSmallKernel, 3, sizeof(cl_uint), (void*)&m_N_padded);
		ocl_check(err, "Failed to set kernel args: MergeSortGlobal");

		for (; stride <= m_N_padded; stride <<= 1) {
			//calculate work sizes
			size_t neededWorkers = m_N_padded / stride;

			localWorkSize[0] = min(LocalWorkSize[0], neededWorkers);
			globalWorkSize[0] = GetGlobalWorkSize(neededWorkers, localWorkSize[0]);

			err = clSetKernelArg(m_MergesortGlobalSmallKernel, 0, sizeof(metrics_GPU), (void*)&metrics_GPU);
			err |= clSetKernelArg(m_MergesortGlobalSmallKernel, 1, sizeof(ordered_metrics_GPU), (void*)&ordered_metrics_GPU);
			err |= clSetKernelArg(m_MergesortGlobalSmallKernel, 2, sizeof(cl_uint), (void*)&stride);
			ocl_check(err, "Failed to set kernel args: m_MergesortGlobalSmallKernel");

			err = clEnqueueNDRangeKernel(CommandQueue, m_MergesortGlobalSmallKernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, &sort_task_evt);
			ocl_check(err, "Error executing kernel m_MergesortGlobalSmallKernel!");

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

			err = clEnqueueNDRangeKernel(CommandQueue, m_MergesortGlobalBigKernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, &sort_task_evt);
			ocl_check(err, "Error executing kernel m_MergesortGlobalBigKernel!");

			if (stride >= 1024 * 1024) ocl_check(clFinish(CommandQueue), "Failed finish CommandQueue at mergesort for bigger strides.");
			swap(metrics_GPU, ordered_metrics_GPU);
		}
	}

	cl_event read_sorted_metrics_evt;
	err = clEnqueueReadBuffer(que, metrics_GPU, CL_TRUE, //turns out this is the right array to read from because the other one is missing the last step of course...
		0, metrics_memsize, ordered_metrics,
		1, &sort_task_evt, &read_sorted_metrics_evt);
	ocl_check(err, "read buffer ordered_metrics_GPU");
	
}



int main(int argc, char *argv[])
{

	if (argc != 2) {
		error("syntax: graph_init datasetName");
	}

	ocl_init("./graph_init.ocl","entry_discover", "compute_metrics");

	//LEGGERE IL DATASET E INIZIALIZZARE LA DAG
	DAG = initDagWithDataSet(argv[1]);
	DAG->Print();
	cout<<"\n";
	//print(DAG->adj, DAG->n, DAG->n, ", ");
	
	int n_entries = entry_discover();
	cout<<"entrypoints: "<<n_entries<<endl;
	print(entrypoints, DAG->n, ", ", true);
	cout<<"\n";

	m_N_padded = round_mul_up(DAG->len,2);
	const int metrics_len = m_N_padded;
	if(metrics_len < (DAG->len)) error("array metrics più piccolo di nodes array");

	//metrics = new cl_int2[DAG->n];
	metrics = new cl_int2[metrics_len]; for (int i = 0; i < metrics_len; i++) { metrics[i].x = i; metrics[i].y = 0; }
	metrics_memsize = metrics_len*sizeof(cl_int2);
	
	visited = new int[DAG->n]; for (int i = 0; i < DAG->n; i++) visited[i] = 0;
	visited_memsize = DAG->n*sizeof(int);
	metrics_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, metrics_memsize, NULL, &err);
	ocl_check(err, "create buffer metrics_GPU");
	ordered_metrics_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, metrics_memsize, NULL, &err);
	ocl_check(err, "create buffer ordered_metrics_GPU");
	cl_event write_visited_evt, write_metrics_evt;
	err = clEnqueueWriteBuffer(que, metrics_GPU, CL_TRUE,
		0, metrics_memsize, metrics,
		0, NULL, &write_metrics_evt);
	ocl_check(err, "write into metrics_GPU");
	compute_metrics(n_entries);
	cout<<"metrics: "<<metrics_len<<endl;
	print(metrics, DAG->n, "\n", true);
	cout<<"\n";

	ordered_metrics = new cl_int2[metrics_len]; for (int i = 0; i < metrics_len; i++) ordered_metrics[i] = metrics[i];
	err = clEnqueueWriteBuffer(que, ordered_metrics_GPU, CL_TRUE,
		0, metrics_memsize, ordered_metrics,
		0, NULL, &write_metrics_evt);
	ocl_check(err, "write into metrics_GPU");
	//sort_task();
	Sort_Mergesort();	
	cout<<"sorted 1: "<<endl;
	print(ordered_metrics, metrics_len, "\n");
	cout<<"\n";

	// err = clEnqueueReadBuffer(que, metrics_GPU, CL_TRUE,
	// 	0, metrics_memsize, metrics,
	// 	1, NULL, NULL);
	// ocl_check(err, "read buffer metrics_GPU");

	// print(metrics, DAG->n);
	// cout<<"\n";

	clReleaseMemObject(queue_GPU);
	clReleaseMemObject(queue_count_GPU);
	clReleaseMemObject(next_queue_GPU);
	clReleaseMemObject(next_queue_count_GPU);
	clReleaseMemObject(graph_edges_GPU);
	clReleaseMemObject(metrics_GPU);
	clReleaseKernel(compute_metrics_k);
	

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
		
	entry_discover_k = clCreateKernel(prog, kernelNameEntryDiscover, &err);
	ocl_check(err, "create kernel %s", kernelNameEntryDiscover);
	compute_metrics_k = clCreateKernel(prog, kernelNameComputeMetrics, &err);
	ocl_check(err, "create kernel %s", kernelNameComputeMetrics);
	// sort_task_k = clCreateKernel(prog2, "Sort_MergesortStart", &err);
	// ocl_check(err, "create kernel %s", "Sort_MergesortStart");

	preferred_wg_size = get_preferred_work_group_size_multiple(compute_metrics_k, que);

	m_MergesortStartKernel = clCreateKernel(prog2, "Sort_MergesortStart", &err);
	ocl_check(err, "create kernel %s", "Sort_MergesortStart");
	m_MergesortGlobalSmallKernel = clCreateKernel(prog2, "Sort_MergesortGlobalSmall", &err);
	ocl_check(err, "create kernel %s", "Sort_MergesortGlobalSmall");
	m_MergesortGlobalBigKernel = clCreateKernel(prog2, "Sort_MergesortGlobalBig", &err);
	ocl_check(err, "create kernel %s", "Sort_MergesortGlobalBig");
}


Graph<int>* initDagWithDataSet(string dataset_file_name){

	ifstream data_set;
	//data_set.open("./data_set/first.txt");
	stringstream ss;
	ss << "./data_set/" << dataset_file_name << ".txt";
	string dataset_file_name_with_extension = ss.str();
	data_set.open(dataset_file_name_with_extension);
	cout<<"dataset_file_name_with_extension: "<<dataset_file_name_with_extension<<endl;
	if(!data_set.is_open()){
	    error("impossibile aprire il dataset");
	}

	cout<<"step 1"<<endl;

	int n_nodes = 0;
	data_set >> n_nodes;
		
	Graph<int> *DAG = new Graph<int>(n_nodes);
	if (!DAG) {
		data_set.close();
		error("failed to allocate graph");
	}

	cout<<"step 2"<<endl;

	//leggo tutti gli id in prima posizione in modo da creare la dag senza adj per il momento.
	int value, n_successor, successor_index, data_transfer;
	while(data_set >> value >> n_successor){
		int index = DAG->insertNode(value);
		for (int i = 0; i < n_successor; i++)
		{
			data_set>>successor_index>>data_transfer;
		}
	}

	cout<<"step 3"<<endl;

	data_set.clear();
	data_set.seekg(0);
	data_set >> n_nodes;

	//adesso leggo di nuovo per creare la adj

	while(data_set >> value >> n_successor){
		for (int i = 0; i < n_successor; i++)
		{
			data_set>>successor_index>>data_transfer;
			DAG->insertEdgeByIndex(DAG->indexOfNode(value), successor_index, 1/*data_transfer*/); //TODO: al momento assumo che l'indice sia l'id dell'elemento, altrimenti avrei dovuto leggere i dati da dataset a partire dal fondo a causa delle dipendenze.
		}
	}

	cout<<"step 4"<<endl;

	//TODO: e se mantenessi in memoria una matrice quadrata con tutti questi dati in fila per ogni task? IN modo da avere coalescenza?

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


