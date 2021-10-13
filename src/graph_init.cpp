#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include "dag.h"
#include "graphnode.h"

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
cl_kernel init_graph_k;
cl_int err;

void ocl_init(char* progName, char* kernelName){
	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	ctx = create_context(p, d);
	que = create_queue(ctx, d);
	cl_program prog = create_program(progName, ctx, d);

	init_graph_k = clCreateKernel(prog, kernelName, &err);

	ocl_check(err, "create kernel init_graph");
}

cl_event init_graph(cl_command_queue q, cl_kernel init_graph_k,
	int n_nodes, int n_processors, cl_mem graph_nodes_GPU, cl_mem graph_edges_GPU)
{
	int arg_index = 0;
	size_t gws[] = { n_nodes };

	cl_int err;
	err = clSetKernelArg(init_graph_k, arg_index++, sizeof(n_nodes), &n_nodes);
	ocl_check(err,  "set arg %d for init_graph_k", arg_index);
	err = clSetKernelArg(init_graph_k, arg_index++, sizeof(n_processors), &n_processors);
	ocl_check(err,  "set arg %d for init_graph_k", arg_index);
	err = clSetKernelArg(init_graph_k, arg_index++, sizeof(graph_nodes_GPU), &graph_nodes_GPU);
	ocl_check(err,  "set arg %d for init_graph_k", arg_index);
	err = clSetKernelArg(init_graph_k, arg_index++, sizeof(graph_edges_GPU), &graph_edges_GPU);
	ocl_check(err,  "set arg %d for init_graph_k", arg_index);

	cl_event init_graph_evt;
	err = clEnqueueNDRangeKernel(q, init_graph_k,
		1, NULL, gws, NULL,
		0, NULL, &init_graph_evt);

	ocl_check(err, "enqueue init_graph");

	return init_graph_evt;
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


	int id, n_successor, successor, data_transfer;
	while(data_set >> id >> n_successor){
		id--;//brutto hack a causa degli indici
		DAG->insertNode(id);
		for (int i = 0; i < n_successor; i++)
		{
			data_set>>successor>>data_transfer;
			successor--;//brutto hack
			DAG->insertEdgeByIndex(DAG->indexOfNode(id), successor, data_transfer); //TODO: al momento assumo che l'indice sia l'id dell'elemento, altrimenti avrei dovuto leggere i dati da dataset a partire dal fondo a causa delle dipendenze.
		}
		
	}

	data_set.close();
	return DAG;	
}

int main(int argc, char *argv[])
{

	if (argc != 2) {
		error("syntax: graph_init numberOfProcessors");
	}
	int n_processors = atoi(argv[1]);
	if (n_processors < 1)
		error("numberOfProcessors < 1");

	ocl_init("./graph_init.ocl","init_graph");

	//LEGGERE IL DATASET E INIZIALIZZARE LA DAG
	Graph<int> *DAG = initDagWithDataSet();

	int const n_nodes = DAG->len;//, n_edges = 10;
	size_t nodes_memsize = n_nodes*sizeof(int);
	size_t edges_memsize = n_nodes*n_nodes*sizeof(int);

	//CREO MEMORIA BUFFER IN GPU
	cl_mem graph_nodes_GPU = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, nodes_memsize, NULL, &err);
	ocl_check(err, "create buffer graph_nodes_GPU");
	cl_mem graph_edges_GPU = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, edges_memsize, NULL, &err);
	ocl_check(err, "create buffer graph_edges_GPU");

	//PASSARE I DATI DELLA DAG ALLA GPU 
	cl_event write_nodes_evt, write_edges_evt;
	err = clEnqueueWriteBuffer(que, graph_nodes_GPU, CL_TRUE,
		0, nodes_memsize, DAG->nodes,
		0, NULL, &write_nodes_evt);
	ocl_check(err, "write dataset nodes into graph_nodes_GPU");
	err = clEnqueueWriteBuffer(que, graph_edges_GPU, CL_TRUE,
		0, edges_memsize, DAG->adj,
		0, NULL, &write_edges_evt);
	ocl_check(err, "write dataset edges into graph_edges_GPU");

	//ESEGUIRE L'ALGORITMO DI SCHEDULING SU GPU
	cl_event init_graph_evt = init_graph(que, init_graph_k, n_nodes, n_processors, graph_nodes_GPU, graph_edges_GPU);

	//CREO I DATI CORRISPONDENTI A QUELLI CHE HO FATTO CREARE ALLA GPU IN CPU
	

	//PASSA I RISULTATI DELLO SCHEDULING DALLA GPU ALLA CPU
	// cl_event read_nodes_evt;
	// err = clEnqueueReadBuffer(que, graph_nodes_GPU, CL_TRUE,
	// 	0, nodes_memsize, DAG->nodes,xx
	// 	1, &init_graph_evt, &read_nodes_evt);
	// ocl_check(err, "read buffer graph_nodes_GPU");

	//METRICHE
	//measurePerformance(init_graph_evt, read_nodes_evt,read_edges_evt, n_nodes, nodes_memsize);
	//VERIFICA DELLA CORRETTEZZA
	//verify(n_nodes, DAG->nodes, DAG->adj);
	//PULIZIA FINALE
	free(DAG);
	clReleaseMemObject(graph_nodes_GPU);
	clReleaseMemObject(graph_edges_GPU);
	clReleaseKernel(init_graph_k);
	
	system("PAUSE");
}