#include <stdio.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"

void error(char const *str)
{
	fprintf(stderr, "%s\n", str);
	exit(1);
}

cl_event init_vec(cl_command_queue q, cl_kernel init_vec_k,
	int nels, cl_mem d_vec)
{
	size_t gws[] = { nels };

	cl_int err;

	err = clSetKernelArg(init_vec_k, 0, sizeof(d_vec), &d_vec);
	ocl_check(err, "set arg 0 for init_vec_k");

	cl_event init_vec_evt;
	err = clEnqueueNDRangeKernel(q, init_vec_k,
		1, NULL, gws, NULL,
		0, NULL, &init_vec_evt);

	ocl_check(err, "enqueue init_vec");

	return init_vec_evt;
}

void verify(int nels, int const *vec) {
	for (int i = 0; i < nels; ++i) {
		if (vec[i] != i) {
			fprintf(stderr, "%d != %d\n", i, vec[i]);
			error("mismatch");
		}
	}
}

int main(int argc, char *argv[])
{
	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	//TODO: posso mettere in path relativo?
	cl_program prog = create_program("C:\\Users\\gabry\\Universita\\GPGPU\\Progetto\\SchedulingWithGPU\\src\\vecinit.ocl", ctx, d);

	cl_int err;
	cl_kernel init_vec_k = clCreateKernel(prog, "init_vec", &err);

	ocl_check(err, "create kernel init_vec");

	if (argc != 2) {
		error("syntax: vecinit_cl nels");
	}

	int nels = atoi(argv[1]);
	if (nels < 0) {
		error("nels < 0");
	}

	size_t memsize = nels*sizeof(int);

	cl_mem d_vec = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, memsize, NULL, &err);
	ocl_check(err, "create buffer d_vec");

	int *h_vec = malloc(memsize);
	if (!h_vec) {
		error("failed to allocate vec");
	}

	cl_event init_evt = init_vec(que, init_vec_k, nels, d_vec);
	cl_event read_evt;

	err = clEnqueueReadBuffer(que, d_vec, CL_TRUE,
		0, memsize, h_vec,
		1, &init_evt, &read_evt);
	ocl_check(err, "read buffer d_vec");

	double runtime_init_ms = runtime_ms(init_evt);
	double runtime_read_ms = runtime_ms(read_evt);

	printf("init: runtime %.4gms, %.4g GE/s, %.4g GB/s\n",
		runtime_init_ms, nels/runtime_init_ms/1.0e6, memsize/runtime_init_ms/1.0e6);
	printf("read: runtime %.4gms, %.4g GE/s, %.4g GB/s\n",
		runtime_read_ms, nels/runtime_read_ms/1.0e6, memsize/runtime_read_ms/1.0e6);


	verify(nels, h_vec);

	free(h_vec);

	clReleaseMemObject(d_vec);
	clReleaseKernel(init_vec_k);
	
	system("PAUSE");
}


