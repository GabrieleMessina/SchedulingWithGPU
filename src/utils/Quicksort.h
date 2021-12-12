/* ************************************************************************* *\
               INTEL CORPORATION PROPRIETARY INFORMATION
     This software is supplied under the terms of a license agreement or 
     nondisclosure agreement with Intel Corporation and may not be copied 
     or disclosed except in accordance with the terms of that agreement. 
        Copyright (C) 2014 Intel Corporation. All Rights Reserved.
\* ************************************************************************* */

#ifndef QUICKSORT_H
#define QUICKSORT_H

int compare (cl_int2 V1, cl_int2 V2){
	if(V1.y == V2.y){
		return (V1.x < V2.x) ? 1 : -1;
	}
	return (V1.y > V2.y) ? 1 : -1;
}

bool operator < (const cl_int2 &V1, const cl_int2 &V2){
	return compare(V1, V2) < 0;
}

bool operator > (const cl_int2 &V1, const cl_int2 &V2){
	return compare(V1, V2) > 0;
}

bool operator <= (const cl_int2 &V1, const cl_int2 &V2){
	return compare(V1, V2) <= 0;
}

bool operator >= (const cl_int2 &V1, const cl_int2 &V2){
	return compare(V1, V2) >= 0;
}

bool operator == (const cl_int2 &V1, const cl_int2 &V2){
	return compare(V1, V2) == 0;
}

bool operator != (const cl_int2 &V1, const cl_int2 &V2){
	return compare(V1, V2) != 0;
}

#ifdef HOST
template <class T>
T median(T x1, T x2, T x3) {
	if (x1 < x2) {
		if (x2 < x3) {
			return x2;
		} else {
			if (x1 < x3) {
				return x3;
			} else {
				return x1;
			}
		}
	} else { // x1 >= x2
		if (x1 < x3) {
			return x1;
		} else { // x1 >= x3
			if (x2 < x3) {
				return x2;
			} else {
				return x3;
			}
		}
	}
}
#else // HOST
uint median(uint x1, uint x2, uint x3) {
	if (x1 < x2) {
		if (x2 < x3) {
			return x2;
		} else {
			if (x1 < x3) {
				return x3;
			} else {
				return x1;
			}
		}
	} else { // x1 >= x2
		if (x1 < x3) {
			return x1;
		} else { // x1 >= x3
			if (x2 < x3) {
				return x2;
			} else {
				return x3;
			}
		}
	}
}
#endif //HOST

#define TRUST_BUT_VERIFY 1
// Note that SORT_THRESHOLD should always be 2X LQSORT_LOCAL_WORKGROUP_SIZE due to the use of bitonic sort
// Always try LQSORT_LOCAL_WORKGROUP_SIZE to be 8X smaller than QUICKSORT_BLOCK_SIZE - then try everything else :)
#ifdef CPU_DEVICE
#define QUICKSORT_BLOCK_SIZE         1536 
#define GQSORT_LOCAL_WORKGROUP_SIZE     4 
#define LQSORT_LOCAL_WORKGROUP_SIZE     4 
#define SORT_THRESHOLD                  8 
#else 
#define QUICKSORT_BLOCK_SIZE         1536 
#define GQSORT_LOCAL_WORKGROUP_SIZE   256 
#define LQSORT_LOCAL_WORKGROUP_SIZE   256 
#define SORT_THRESHOLD                512 
#endif

#define EMPTY_RECORD                   42

// work record contains info about the part of array that is still longer than QUICKSORT_BLOCK_SIZE and 
// therefore cannot be processed by lqsort_kernel yet. It contins the start and the end indexes into 
// an array to be sorted, associated pivot and direction of the sort. 
typedef struct work_record {
	uint start;
	uint end;
	uint pivot;
	uint direction;
#ifdef HOST
	work_record() : 
		start(0), end(0), pivot(0), direction(EMPTY_RECORD) {}
	work_record(uint s, uint e, uint p, uint d) : 
		start(s), end(e), pivot(p), direction(d) {}
#endif // HOST
} work_record;


// parent record contains everything kernels need to know about the parent of a set of blocks:
// initially, the first two fields equal to the third and fourth fields respectively
// blockcount contains the total number of blocks associated with the parent.
// During processing, sstart and send get incremented. At the end of gqsort_kernel, all the 
// parent record fields are used to calculate new pivots and new work records.
typedef struct parent_record {
	uint sstart, send, oldstart, oldend, blockcount; 
#ifdef HOST
	parent_record(uint ss, uint se, uint os, uint oe, uint bc) : 
		sstart(ss), send(se), oldstart(os), oldend(oe), blockcount(bc) {}
#endif // HOST
} parent_record;

// block record contains everything kernels needs to know about the block:
// start and end indexes into input array, pivot, direction of sorting and the parent record index
typedef struct block_record {
	uint start;
	uint end;
	uint pivot;
	uint direction;
	uint parent;
#ifdef HOST
	block_record() : start(0), end(0), pivot(0), direction(EMPTY_RECORD), parent(0) {}
	block_record(uint s, uint e, uint p, uint d, uint prnt) : 
		start(s), end(e), pivot(p), direction(d), parent(prnt) {}
#endif // HOST
} block_record;
#endif // QUICKSORT_H
