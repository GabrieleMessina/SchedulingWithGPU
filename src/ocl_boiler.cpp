
#pragma warning(disable:4996)

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "app_globals.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <iostream>

#define BUFSIZE 4096
char platform_name[BUFSIZE];
int platform_number = 0;
char device_name[BUFSIZE];
int device_number = 0;

void ocl_check(cl_int err, const char* msg, ...) {
	if (err != CL_SUCCESS) {
		char msg_buf[BUFSIZE + 1];
		va_list ap;
		va_start(ap, msg);
		vsnprintf(msg_buf, BUFSIZE, msg, ap);
		va_end(ap);
		msg_buf[BUFSIZE] = '\0';
		fprintf(stderr, "%s - error %d\n", msg_buf, err);
		exit(1);
	}
}

size_t GetGlobalWorkSize(size_t DataElemCount, size_t LocalWorkSize) {
	size_t r = DataElemCount % LocalWorkSize;
	if (r == 0)
		return DataElemCount;
	else
		return DataElemCount + LocalWorkSize - r;
}

void CheckCLError(cl_int err, const char* msg, ...) {
	if (err != CL_SUCCESS) {
		char msg_buf[BUFSIZE + 1];
		va_list ap;
		va_start(ap, msg);
		vsnprintf(msg_buf, BUFSIZE, msg, ap);
		va_end(ap);
		msg_buf[BUFSIZE] = '\0';
		fprintf(stderr, "%s - error %d\n", msg_buf, err);
		exit(1);
	}
}

char* getSelectedPlatformInfo(int& platform_number_out) {
	platform_number_out = platform_number;
	return platform_name;
}


// Return the ID of the platform specified in the OCL_PLATFORM
// environment variable (or the first one if none specified)
cl_platform_id select_platform() {
	cl_uint nplats;
	cl_int err;
	cl_platform_id* plats;

	err = clGetPlatformIDs(0, NULL, &nplats); //46MB
	ocl_check(err, "counting platforms");

	if (DEBUG_OCL_INIT)
		printf("number of platforms: %u\n", nplats);

	plats = (cl_platform_id*)malloc(nplats * sizeof(*plats));

	err = clGetPlatformIDs(nplats, plats, NULL);
	ocl_check(err, "getting platform IDs");

	char buff[BUFSIZE];
	FILE* config = fopen("./utils/config.txt", "a+");
	const char* const env = fgets(buff, 2, config);
	fclose(config);

	if (buff && buff[0] != '\0')platform_number = atoi(buff);

	if (platform_number >= nplats) {
		fprintf(stderr, "no platform number %u", platform_number);
		exit(1);
	}

	cl_platform_id choice = plats[platform_number];

	err = clGetPlatformInfo(choice, CL_PLATFORM_NAME, BUFSIZE,
		platform_name, NULL);
	ocl_check(err, "getting platform name");

	if (DEBUG_OCL_INIT)
		printf("selected platform %d: %s\n", platform_number, platform_name);

	free(plats);
	return choice;
}

char* getSelectedDeviceInfo(int& device_number_out) {
	device_number_out = device_number;
	return device_name;
}

// Return the ID of the device (of the given platform p) specified in the
// OCL_DEVICE environment variable (or the first one if none specified)
cl_device_id select_device(cl_platform_id p) {
	cl_uint ndevs;
	cl_int err;
	cl_device_id* devs;
	const char* const env = getenv("OCL_DEVICE");
	cl_uint device_number = 0;
	if (env && env[0] != '\0')
		device_number = atoi(env);

	err = clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, NULL, &ndevs);
	ocl_check(err, "counting devices");

	if (DEBUG_OCL_INIT)
		printf("number of devices: %u\n", ndevs);

	devs = (cl_device_id*)malloc(ndevs * sizeof(*devs));

	err = clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, ndevs, devs, NULL);
	ocl_check(err, "devices #2");

	if (device_number >= ndevs) {
		fprintf(stderr, "no device number %u", device_number);
		exit(1);
	}

	cl_device_id choice = devs[device_number];

	err = clGetDeviceInfo(choice, CL_DEVICE_NAME, BUFSIZE,
		device_name, NULL);
	ocl_check(err, "device name");

	printf("selected device %d: %s\n", device_number, device_name);

	free(devs);
	return choice;
}

// Create a one-device context
cl_context create_context(cl_platform_id p, cl_device_id d) {
	cl_int err;

	cl_context_properties ctx_prop[] = {
		CL_CONTEXT_PLATFORM, (cl_context_properties)p, 0
	};

	cl_context ctx = clCreateContext(ctx_prop, 1, &d, //83MB
		NULL, NULL, &err);
	ocl_check(err, "create context");

	return ctx;
}

// Create a command queue for the given device in the given context
cl_command_queue create_queue(cl_context ctx, cl_device_id d) {
	cl_int err;

	cl_command_queue que = clCreateCommandQueue(ctx, d,
		CL_QUEUE_PROFILING_ENABLE, &err);
	ocl_check(err, "create queue");
	return que;
}

// Compile the device part of the program, stored in the external
// file `fname`, for device `dev` in context `ctx`
cl_program create_program(const char* const fname, cl_context ctx,
	cl_device_id dev) {
	cl_int err, errlog;
	cl_program prg;

	char src_buf[BUFSIZE + 1];
	char* log_buf = NULL;
	size_t logsize;
	const char* buf_ptr = src_buf;
	time_t now = time(NULL);

	memset(src_buf, 0, BUFSIZE);

	snprintf(src_buf, BUFSIZE, "// %s#include \"%s\"\n",
		ctime(&now), fname);
	
	if (DEBUG_OCL_INIT)
		printf("compiling:\n%s", src_buf);
	
	prg = clCreateProgramWithSource(ctx, 1, &buf_ptr, NULL, &err);
	ocl_check(err, "create program");

	err = clBuildProgram(prg, 1, &dev, "-I.", NULL, NULL); //165MB
	errlog = clGetProgramBuildInfo(prg, dev, CL_PROGRAM_BUILD_LOG,
		0, NULL, &logsize);
	ocl_check(errlog, "get program build log size");
	log_buf = (char*)malloc(logsize);
	errlog = clGetProgramBuildInfo(prg, dev, CL_PROGRAM_BUILD_LOG,
		logsize, log_buf, NULL);
	ocl_check(errlog, "get program build log");
	while (logsize > 0 &&
		(log_buf[logsize - 1] == '\n' ||
			log_buf[logsize - 1] == '\0')) {
		logsize--;
	}
	if (logsize > 0) {
		log_buf[logsize] = '\n';
		log_buf[logsize + 1] = '\0';
	}
	else {
		log_buf[logsize] = '\0';
	}

	if (DEBUG_OCL_INIT)
		printf("=== BUILD LOG ===\n%s\n=========\n", log_buf);
	ocl_check(err, "build program");

	if(log_buf != NULL)free(log_buf);
	return prg;
}

size_t get_preferred_work_group_size_multiple(cl_kernel k, cl_command_queue q) {
	size_t wg_mul;
	cl_device_id d;
	cl_int err = clGetCommandQueueInfo(q, CL_QUEUE_DEVICE, sizeof(d), &d, NULL);
	ocl_check(err, "get command queue device");
	err = clGetKernelWorkGroupInfo(k, d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(wg_mul), &wg_mul, NULL);
	ocl_check(err, "get preferred work-group size multiple");
	return wg_mul;
}

// Runtime of an event, in nanoseconds. Note that if NS is the
// runtime of an event in nanoseconds and NB is the number of byte
// read and written during the event, NB/NS is the effective bandwidth
// expressed in GB/s
cl_ulong runtime_ns(cl_event evt) {
	cl_int err;
	cl_ulong start, end;
	err = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START,
		sizeof(start), &start, NULL);
	ocl_check(err, "get start");
	err = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END,
		sizeof(end), &end, NULL);
	ocl_check(err, "get end");
	return (end - start);
}

cl_ulong total_runtime_ns(cl_event from, cl_event to) {
	cl_int err;
	cl_ulong start, end;
	err = clGetEventProfilingInfo(from, CL_PROFILING_COMMAND_START,
		sizeof(start), &start, NULL);
	ocl_check(err, "get start");
	err = clGetEventProfilingInfo(to, CL_PROFILING_COMMAND_END,
		sizeof(end), &end, NULL);
	ocl_check(err, "get end");
	return (end - start);
}

// Runtime of an event, in milliseconds
double runtime_ms(cl_event evt) {
	return runtime_ns(evt) * 1.0e-6;
}

double total_runtime_ms(cl_event from, cl_event to) {
	return total_runtime_ns(from, to) * 1.0e-6;
}

/* round gws to the next multiple of lws */
size_t round_mul_up(size_t gws, size_t lws) {
	return ((gws + lws - 1) / lws) * lws;
}

// size_t getPaddedSize(size_t n)
// {
// 	unsigned int log2val = (unsigned int)ceil(log((float)n) / log(2.f));
// 	return (size_t)pow(2, log2val);
// }