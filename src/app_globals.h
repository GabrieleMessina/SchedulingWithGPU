#pragma once

/*ocl*/
#define CL_TARGET_OPENCL_VERSION 120
/*ocl*/

/*debug*/
#define DEBUG_MEMORY_LEAK false
#define DEBUG_OCL_INIT false
#define DEBUG_DAG_INIT false
#define DEBUG_ENTRY_DISCOVER false
#define DEBUG_COMPUTE_METRICS false
#define DEBUG_SORT false
#define DEBUG_PROCESSOR_ASSIGNMENT true
#define DEBUG_OCL_METRICS true
#define DEBUG_HEAP_ALLOC false
/*debug*/

/*DAG*/
#define edge_t int
#define metrics_t cl_int3
#define metrics_tt int3 //for kernels
#define VECTOR_ADJ false
#define TRANSPOSED_ADJ false
#define RECTANGULAR_ADJ true
/*DAG*/

/*utils*/
#ifndef NULL
    #define NULL 0
#endif

#ifdef _WIN32
#define WINDOWS true
#else 
#define WINDOWS false
#endif

#if DEBUG_HEAP_ALLOC
    #define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
    // Replace _NORMAL_BLOCK with _CLIENT_BLOCK if you want the
    // allocations to be of _CLIENT_BLOCK type
#else
    #define DBG_NEW new
#endif
/*utils*/
