#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <chrono>
#include <ctime>   
#include "dag.h"
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
cl_kernel entry_discover_k, compute_metrics_k, /*sort_task_k,*/ m_MergesortGlobalBigKernel, m_MergesortGlobalSmallKernel, m_MergesortStartKernel;
void ocl_init(char *progName, char *kernelNameEntryDiscover, char *kernelNameComputeMetrics);
size_t preferred_wg_size;

Graph<int> *DAG;
Graph<int>* initDagWithDataSet(string fileName);


void measurePerformance(cl_event entry_discover_evt,cl_event compute_metrics_evt, cl_event sort_task_evt, int nels);
void measurePerformance(cl_event evt, int nels, string event_name);
void verify();

int *entrypoints; 
size_t m_N_padded;
cl_event _entry_discover(int n_nodes, cl_mem graph_edges_GPU, cl_mem n_entrypoints_GPU, cl_mem entrypoints_GPU)
{
	int arg_index = 0;
	size_t gws[] = { n_nodes }; //TODO: round mul up

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
		1, NULL, gws, NULL,
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
	entrypoints = new int[n_nodes]; for (int i = 0; i < n_nodes; i++) entrypoints[i] = -1;
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
	//size_t lws[] = { n_nodes * 2 }; //number of work item to launch in the same work group, that means they are garanted to be executed at the same time.
	//size_t gws[] = {GetGlobalWorkSize(n_nodes, lws[0])}; //total work item to launch.
	const size_t gws[] = { n_nodes }; //TODO: round mul up
	//const size_t gws[] = { round_mul_up(n_nodes, preferred_wg_size) };

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

cl_event compute_metrics(){
	int const n_nodes = DAG->len;
	const int metrics_len = m_N_padded;
	queue = entrypoints;
	next_queue = new int[n_nodes]; for (int i = 0; i < n_nodes; i++) next_queue[i] = -1;
	nodes_memsize = n_nodes*sizeof(int);
	queue_memsize = n_nodes*sizeof(int);
	edges_memsize = n_nodes*n_nodes*sizeof(bool);
	metrics_memsize = metrics_len*sizeof(cl_int2);

	//Creo e inizializzo memoria host e device per array che conterrà le metriche.
	if(metrics_len < (DAG->len)) error("array metrics più piccolo di nodes array");
	metrics = new cl_int2[metrics_len]; for (int i = 0; i < metrics_len; i++) { metrics[i].x = i; metrics[i].y = 0; }
	

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

	cl_event compute_metrics_evt = _compute_metrics(graph_nodes_GPU, queue_GPU, next_queue_GPU, graph_edges_GPU, metrics_GPU);
	
	//se non vogliamo farlo ad ogni ciclo, dovremmo mantenere un bool nel kernel che viene settato a true appena qualcuno scrive in next_queue, quindi invece di tutto next_queue possiamo leggere ad ogni ciclo solo il bit.
	cl_event read_next_queue_evt, read_metrics_evt;
	err = clEnqueueReadBuffer(que, next_queue_GPU, CL_TRUE, 0, queue_memsize, next_queue, 1, &compute_metrics_evt, &read_next_queue_evt);
	ocl_check(err, "read buffer next_queue_GPU");

	while(!isEmpty(next_queue, n_nodes, -1)) {
		//scambio le due code e reinizializzo la next_queue in modo da poter essere riutilizzata.
		queue_GPU = next_queue_GPU;
		next_queue = new int[n_nodes]; for (int i = 0; i < n_nodes; i++) next_queue[i] = -1;

		//passo il next_array reinizializzato alla GPU.
		next_queue_GPU = clCreateBuffer(ctx, CL_MEM_READ_WRITE, queue_memsize, NULL, &err);
		ocl_check(err, "create buffer next_queue_GPU");
		err = clEnqueueWriteBuffer(que, next_queue_GPU, CL_TRUE, 0, queue_memsize, next_queue,0, NULL, &write_next_queue_evt);
		ocl_check(err, "write into next_queue_GPU");

		compute_metrics_evt = _compute_metrics(graph_nodes_GPU, queue_GPU, next_queue_GPU, graph_edges_GPU, metrics_GPU);

		//leggo il nuovo next_array dalla GPU.
		err = clEnqueueReadBuffer(que, next_queue_GPU, CL_TRUE, 0, queue_memsize, next_queue, 1, &compute_metrics_evt, &read_next_queue_evt);
		ocl_check(err, "read buffer next_queue_GPU");
	}

	//PASSA I RISULTATI DALLA GPU ALLA CPU
	err = clEnqueueReadBuffer(que, metrics_GPU, CL_TRUE,
		0, metrics_memsize, metrics,
		1, &compute_metrics_evt, &read_metrics_evt);
	ocl_check(err, "read buffer metrics_GPU");

	//PULIZIA FINALE
	clReleaseMemObject(queue_GPU);
	clReleaseMemObject(queue_count_GPU);
	clReleaseMemObject(next_queue_GPU);
	clReleaseMemObject(next_queue_count_GPU);
	clReleaseMemObject(graph_edges_GPU);

	clReleaseKernel(compute_metrics_k);

	return compute_metrics_evt;
}

#define MERGESORT_SMALL_STRIDE 1024 
cl_event Sort_Mergesort()
{ 
	int c = 0;
	const int metrics_len = m_N_padded;
	cl_event sort_task_evt;
	cl_context Context = ctx; 
	cl_command_queue CommandQueue = que;
	size_t LocalWorkSize[3] = { preferred_wg_size,1,1 }; //TODO:il lws necessita di essere un multiplo di m_N_padded, suppongo a causa dell'implementazione, è un problema?
	//TODO fix memory problem when many elements. -> CL_OUT_OF_RESOURCES
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

	// proceed with the global variant
	unsigned int stride = 2 * locLimit;
	if (m_N_padded <= MERGESORT_SMALL_STRIDE) {
		//non funziona per grandi numeri perché crea dei buffer locali e si rischia un cl out of resources.
		cout<<"small kernel " << (int)m_N_padded<<endl;
		// set not changing arguments
		err = clSetKernelArg(m_MergesortGlobalSmallKernel, 3, sizeof(cl_uint), (void*)&m_N_padded);
		ocl_check(err, "Failed to set kernel args: MergeSortGlobal");

		for (; stride <= m_N_padded; stride <<= 1) { // crea i branch per il merge sort.
			//calculate work sizes
			size_t neededWorkers = m_N_padded / stride;

			localWorkSize[0] = min(LocalWorkSize[0], neededWorkers);
			globalWorkSize[0] = GetGlobalWorkSize(neededWorkers, localWorkSize[0]);

			err = clSetKernelArg(m_MergesortGlobalSmallKernel, 0, sizeof(metrics_GPU), (void*)&metrics_GPU);
			err |= clSetKernelArg(m_MergesortGlobalSmallKernel, 1, sizeof(ordered_metrics_GPU), (void*)&ordered_metrics_GPU);
			err |= clSetKernelArg(m_MergesortGlobalSmallKernel, 2, sizeof(cl_uint), (void*)&stride);
			ocl_check(err, "Failed to set kernel args: m_MergesortGlobalSmallKernel");

			err = clEnqueueNDRangeKernel(CommandQueue, m_MergesortGlobalSmallKernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, &sort_task_evt);
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

			err = clEnqueueNDRangeKernel(CommandQueue, m_MergesortGlobalBigKernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, &sort_task_evt);
			ocl_check(err, "Error executing kernel m_MergesortGlobalBigKernel!");

			//TODO: check this if
			if (stride >= 1024 * 1024) ocl_check(clFinish(CommandQueue), "Failed finish CommandQueue at mergesort for bigger strides.");
			swap(metrics_GPU, ordered_metrics_GPU);
		}
	}

	cl_event read_sorted_metrics_evt;
	err = clEnqueueReadBuffer(que, metrics_GPU, CL_TRUE, //turns out this is the right array to read from because the other one is missing the last step of course...
		0, metrics_memsize, ordered_metrics,
		1, &sort_task_evt, &read_sorted_metrics_evt);
	ocl_check(err, "read buffer ordered_metrics_GPU");

	clReleaseMemObject(metrics_GPU);
	clReleaseMemObject(ordered_metrics_GPU);
	clReleaseKernel(m_MergesortStartKernel);
	clReleaseKernel(m_MergesortGlobalSmallKernel);
	clReleaseKernel(m_MergesortGlobalBigKernel);

	return sort_task_evt;
}

std::chrono::system_clock::time_point start_time;
std::chrono::system_clock::time_point end_time;

int main(int argc, char *argv[])
{

	if (argc != 2) {
		error("syntax: graph_init datasetName");
	}

	start_time = std::chrono::system_clock::now();

	ocl_init("./graph_init.ocl","entry_discover", "compute_metrics");

	//LEGGERE IL DATASET E INIZIALIZZARE LA DAG
	DAG = initDagWithDataSet(argv[1]);
	printf("DAG initialized\n");
	m_N_padded = round_mul_up(DAG->len,2);
	//DAG->Print();
	//cout<<"\n";
	//print(DAG->adj, DAG->n, DAG->n, ", ");
	
	cl_event entry_discover_evt = entry_discover();
	printf("entries discovered\n");
	// cout<<"entrypoints: "<<n_entries<<endl;
	// print(entrypoints, DAG->n, ", ", true);
	// cout<<"\n";

	const int metrics_len = m_N_padded;
	cl_event compute_metrics_evt = compute_metrics();
	printf("metrics computed\n");

	//cout<<"metrics: "<<metrics_len<<endl;
	//print(metrics, DAG->n, "\n", true);
	//cout<<"\n";

	cl_event sort_task_evt = Sort_Mergesort();	
	printf("array sorted\n");

	//cout<<"sorted: "<<endl;
	//print(ordered_metrics, metrics_len);
	//print(ordered_metrics, metrics_len, "\n");
	//cout<<"\n";
	
	end_time = std::chrono::system_clock::now();

	//METRICHE
	measurePerformance(entry_discover_evt, sort_task_evt,entry_discover_evt, DAG->len);
	//VERIFICA DELLA CORRETTEZZA
	verify();

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


void measurePerformance(cl_event entry_discover_evt,cl_event compute_metrics_evt, cl_event sort_task_evt, int nels){
	double runtime_discover_ms = runtime_ms(entry_discover_evt);
	double runtime_metrics_ms = runtime_ms(compute_metrics_evt);
	double runtime_sorts_ms = runtime_ms(sort_task_evt);

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

	double total_elapsed_time_GPU = total_runtime_ms(entry_discover_evt, sort_task_evt);
	int platform_id = 0;
	char *platform_name = getSelectedPlatformInfo(platform_id);
	int device_id = 0;
	char *device_name = getSelectedDeviceInfo(device_id);
	
	//stampo i file in un .csv per poter analizzare i dati successivamente.
	FILE *fp;
	fp = fopen("execution_results.txt", "a");
	if (fp == NULL) {
		printf("Error opening file!\n");
		exit(1);
	}
	
	fprintf(fp,"%s, %f, %f, %s, %s, %.4g, %.4g, %.4g, %d, %d\n", 
		end_date_time, elapsed_seconds.count(), total_elapsed_time_GPU, platform_name, device_name, runtime_discover_ms, runtime_metrics_ms, runtime_sorts_ms, nels, preferred_wg_size);
	
	fflush(fp);
	fclose(fp);
}

void measurePerformance(cl_event evt, int nels, string event_name = "event"){
	double evt_ms = runtime_ms(evt);
	printf("evt %s: runtime %.4gms, %.4g GE/s, %.4g GB/s\n",
		event_name, evt_ms, nels/evt_ms/1.0e6, preferred_wg_size/evt_ms/1.0e6);
}

bool operator<(const cl_int2& l, const cl_int2& r)
{
    if(l.y == r.y){
		return l.x < r.x;
	}
	return l.y > r.y;
}
bool operator> (const cl_int2& l, const cl_int2& r){ return (r < l); }
bool operator<=(const cl_int2& l, const cl_int2& r){ return !(l > r); }
bool operator>=(const cl_int2& l, const cl_int2& r){ return !(l < r); }

void verify() {
	//scandire ordered_metrics e verificare che sia ordinati
	for (int i = 0; i < DAG->len - 1; ++i) {
		if(ordered_metrics[i] < ordered_metrics[i+1]){
			fprintf(stderr, "ordered_metrics[%d] = (%d, %d) < ordered_metrics[%d] = (%d, %d)\n", i, ordered_metrics[i].x, ordered_metrics[i].y, i+1, ordered_metrics[i+1].x, ordered_metrics[i+1].y);
			error("mismatch");
		}
	}
	printf("Everything sorted, verified\n");
}


