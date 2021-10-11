#include <stdio.h>
#include <stdlib.h>
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
	int nels, cl_mem d_graph)
{
	size_t gws[] = { nels };

	cl_int err;

	err = clSetKernelArg(init_graph_k, 0, sizeof(d_graph), &d_graph);
	ocl_check(err, "set arg 0 for init_graph_k");

	cl_event init_graph_evt;
	err = clEnqueueNDRangeKernel(q, init_graph_k,
		1, NULL, gws, NULL,
		0, NULL, &init_graph_evt);

	ocl_check(err, "enqueue init_graph");

	return init_graph_evt;
}

void verify(int nels, int const *vec) {
	for (int i = 0; i < nels; ++i) {
		if (vec[i] != i) {
			fprintf(stderr, "%d != %d\n", i, vec[i]);
			error("mismatch");
		}
	}
}

void measurePerformance(cl_event init_evt, cl_event read_evt, int nels, size_t memsize){
	double runtime_init_ms = runtime_ms(init_evt);
	double runtime_read_ms = runtime_ms(read_evt);

	printf("init: runtime %.4gms, %.4g GE/s, %.4g GB/s\n",
		runtime_init_ms, nels/runtime_init_ms/1.0e6, memsize/runtime_init_ms/1.0e6);
	printf("read: runtime %.4gms, %.4g GE/s, %.4g GB/s\n",
		runtime_read_ms, nels/runtime_read_ms/1.0e6, memsize/runtime_read_ms/1.0e6);
}

int main(int argc, char *argv[])
{
	if (argc != 2) {
		error("syntax: graph_init nels");
	}

	int const nels = atoi(argv[1]);
	if (nels < 0) {
		error("nels < 0");
	}

	ocl_init(
		"C:\\Users\\gabry\\Universita\\GPGPU\\Progetto\\SchedulingWithGPU\\src\\graph_init.ocl", //TODO: posso mettere in path relativo?
		"init_graph"
	);

	size_t memsize = nels*sizeof(int);

	//buffer in GPU
	cl_mem d_graph = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, memsize, NULL, &err);
	ocl_check(err, "create buffer d_graph");

	//dati in CPU
	//TODO: forse la classe grafo è superflua e serve solo passare come parametri i nodi e la matrice di adiacenza per gli edge, potrebbero anche esserci altre matrici di adiacenza per ad esempio le dipendenze o altro visto che le dipendenze sono date proprio dagli edge
	Graph<int> *h_graph = new Graph<int>(nels); //TODO: Graph<GraphNode> anche se servono dei tipi primitivi, quindi da vedere
	if (!h_graph) {
		error("failed to allocate graph");
	}

	cl_event init_evt = init_graph(que, init_graph_k, nels, d_graph);
	cl_event read_evt;

	int* nodes = h_graph->nodes;

	//unisce passa i dati dalla CPU alla GPU
	err = clEnqueueReadBuffer(que, d_graph, CL_TRUE,
		0, memsize, h_graph->nodes,
		1, &init_evt, &read_evt);
	ocl_check(err, "read buffer d_graph");


	//metriche
	measurePerformance(init_evt, read_evt, nels, memsize);
	//verifica della correttezza
	verify(nels, h_graph->nodes);
	//pulizia finale
	free(h_graph);
	clReleaseMemObject(d_graph);
	clReleaseKernel(init_graph_k);
	
	system("PAUSE");
}