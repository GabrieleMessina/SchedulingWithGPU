/* ************************************************************************* *\
               INTEL CORPORATION PROPRIETARY INFORMATION
     This software is supplied under the terms of a license agreement or 
     nondisclosure agreement with Intel Corporation and may not be copied 
     or disclosed except in accordance with the terms of that agreement. 
        Copyright (C) 2014 Intel Corporation. All Rights Reserved.
\* ************************************************************************* */

#include "Quicksort.h"

/// bitonic_sort: sort 2*LQSORT_LOCAL_WORKGROUP_SIZE elements
void bitonic_sort(local uint* sh_data, const uint localid) 
{
	for (uint ulevel = 1; ulevel < LQSORT_LOCAL_WORKGROUP_SIZE; ulevel <<= 1) {
        for (uint j = ulevel; j > 0; j >>= 1) {
            uint pos = 2*localid - (localid & (j - 1));

			uint direction = localid & ulevel;
			uint av = sh_data[pos], bv = sh_data[pos + j];
			const bool sortThem = av > bv;
			const uint greater = select(bv, av, sortThem);
			const uint lesser  = select(av, bv, sortThem);

			sh_data[pos]     = select(lesser, greater, direction);
			sh_data[pos + j] = select(greater, lesser, direction);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

	for (uint j = LQSORT_LOCAL_WORKGROUP_SIZE; j > 0; j >>= 1) {
        uint pos = 2*localid - (localid & (j - 1));

		uint av = sh_data[pos], bv = sh_data[pos + j];
		const bool sortThem = av > bv;
		sh_data[pos]      = select(av, bv, sortThem);
		sh_data[pos + j]  = select(bv, av, sortThem);

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

void sort_threshold(local uint* data_in, global uint* data_out,
					uint start, 
					uint end, local uint* temp, uint localid) 
{
	uint tsum = end - start;
	if (tsum == SORT_THRESHOLD) {
		bitonic_sort(data_in+start, localid);
		for (uint i = localid; i < SORT_THRESHOLD; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
			data_out[start + i] = data_in[start + i];
		}
	} else if (tsum > 1) {
		for (uint i = localid; i < SORT_THRESHOLD; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
			if (i < tsum) {
				temp[i] = data_in[start + i];
			} else {
				temp[i] = UINT_MAX;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		bitonic_sort(temp, localid);

		for (uint i = localid; i < tsum; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
			data_out[start + i] = temp[i];
		}
	} else if (tsum == 1 && localid == 0) {
		data_out[start] = data_in[start];
	} 
}
//#define PROFILE_ME 1
#ifdef PROFILE_ME
global ulong mytimes[2];
#endif

// forward declarations
kernel void gqsort_kernel(global uint* d, global uint* dn, 
						  global block_record* blocks, 
						  global parent_record* parents, 
						  global work_record* result,
						  global work_record* work, 
						  global work_record* done,
						  uint   done_size,
						  uint MAXSEQ,
						  uint num_workgroups);

kernel void lqsort_kernel(global uint* d, global uint* dn, global work_record* seqs);

//---------------------------------------------------------------------------------------
// Kernel implements logic to sort result records into work and done.
// This kernel is launched initially to divide the input sequence into a set of blocks.
// After that it launches gqsort_kernel. The execution alternates between gqsort_kernel
// and relauncher_kernel until all the records are small enough to be processed by the 
// lqsort_kernel. Note two pieces of functionality: sorting results into work set and 
// done set and subdividing work set into blocks.
//
// d - input array
// dn - scratch array of the same size as the input array
// blocks - all the input is split into blocks; each work group is partitioning each 
//          block of data around the pivot
// parents - array of parents associated with different blocks
// result  - new subarrays generated after sorting around a pivot
// work    - used for accumulating records that
//           still need processing by gqsort_kernel
// done    - used for accumulating records that 
//           are small enough to be processed by lqsort_kernel
// done_size - size of the done array
// MAXSEQ  - maximum number of sequences limits
//           the number of blocks passed to gqsort_kernel for processing
// num_workgroups - parameter used to calculate the number of records in the results set.
//---------------------------------------------------------------------------------------
kernel void relauncher_kernel(global uint* d, global uint* dn, 
						  global block_record* blocks, 
						  global parent_record* parents, 
						  global work_record* result,
						  global work_record* work, 
						  global work_record* done,
						  uint   done_size,
						  uint MAXSEQ,
						  uint num_workgroups) 
{
	queue_t q = get_default_queue();
	uint work_size = 0;
	global work_record* it = result;
	for(uint i = 0; i < 2 * num_workgroups; i++, it++) {
		uint direction = it->direction;
		if (direction != EMPTY_RECORD) {
			uint start = it->start;
			uint end = it->end;
			uint block_size = end - start;

			if (block_size <= QUICKSORT_BLOCK_SIZE) {
				if (block_size > 0) {
					done[done_size] = *it;
					done_size++;
				}
			} else {
				work[work_size] = *it;
				work_size++;
			}
			it->direction = EMPTY_RECORD;
		} 
	}

	if (work_size != 0) {
		// Calculate block size, parents and blocks
		uint blocksize = 0;
		global work_record* it = work;
		for(uint i = 0; i < work_size; i++, it++) {
			uint recsize = (it->end - it->start)/MAXSEQ;
			if (recsize == 0)
				recsize = 1;
			blocksize += recsize;
		}

		uint parents_size = 0, blocks_size = 0;
		it = work;
		for(uint i = 0; i < work_size; i++, it++) {
			uint start = it->start;
			uint end   = it->end;
			uint pivot = it->pivot;
			uint direction = it->direction;	
			uint blockcount = (end - start + blocksize - 1)/blocksize;
			if (blockcount == 0)
				blockcount = 1;

			parent_record prnt = {start, end, start, end, blockcount - 1};
			parents[i] = prnt;
			parents_size++;

			for(uint j = 0; j < prnt.blockcount; j++) {
				uint bstart = start + blocksize*j;
				block_record br = {bstart, bstart+blocksize, pivot, direction, parents_size - 1};
				blocks[blocks_size] = br;
				blocks_size++;
			}
			block_record br = {start + blocksize*prnt.blockcount, end, pivot, direction, parents_size - 1};
			blocks[blocks_size] = br;
			blocks_size++;
		}

		enqueue_kernel(q, CLK_ENQUEUE_FLAGS_NO_WAIT, 
			ndrange_1D(GQSORT_LOCAL_WORKGROUP_SIZE * blocks_size, GQSORT_LOCAL_WORKGROUP_SIZE),
			^{ gqsort_kernel(d, dn, blocks, parents, result, work, done, done_size, MAXSEQ, 0); });
	} else {
#ifdef PROFILE_ME
		clk_event_t evt0;
#endif
		enqueue_kernel(q, CLK_ENQUEUE_FLAGS_NO_WAIT, 
			ndrange_1D(LQSORT_LOCAL_WORKGROUP_SIZE * done_size, LQSORT_LOCAL_WORKGROUP_SIZE), 
#ifdef PROFILE_ME
			0, NULL, &evt0,
#endif
			^{ lqsort_kernel(d, dn, done); });
#ifdef PROFILE_ME
		capture_event_profiling_info(evt0, CLK_PROFILING_COMMAND_EXEC_TIME, mytimes);

		printf("\nlqsort_kernel Execution time in milliseconds = %ld ms, %ld ns\n", mytimes[1] / 1000000);
		release_event(evt0);
#endif
	}
}

//------------------------------------------------------------------------------------
// Kernel implements gqsort_kernel
// this kernel is launched repeatedly until all the input is split into small
// enough chunks (all chunks are less than QUICKSORT_BLOCK_SIZE) at which point
// lqsort_kernel is launched. 
// d - input array
// dn - scratch array of the same size as the input array
// blocks - all the input is split into blocks; each work group is partitioning each 
//          block of data around the pivot
// parents - array of parents associated with different blocks
// result  - new subarrays generated after sorting around a pivot
// work    - array is passed to relauncher_kernel: used for accumulating records that
//           still need processing by gqsort_kernel
// done    - array is passed to relauncher_kernel: used for accumulating records that 
//           are small enough to be processed by lqsort_kernel
// done_size - size of the done array
// MAXSEQ  - parameter passed to relauncher_kernel: maximum number of sequences limits
//           the number of blocks passed to gqsort_kernel for processing
// num_workgroups - dummy parameter introduced to match the number of parameters to
//                  the relauncher_kernel - BUG in the compiler.
//------------------------------------------------------------------------------------
kernel void gqsort_kernel(global uint* d, global uint* dn, 
						  global block_record* blocks, 
						  global parent_record* parents, 
						  global work_record* result,
						  global work_record* work, 
						  global work_record* done,
						  uint   done_size,
						  uint MAXSEQ,
						  uint num_workgroups) // BUG - num_workgroups is unneccesary
{
	const uint blockid    = get_group_id(0);
	const uint localid    = get_local_id(0);
	local uint ltsum, gtsum, lbeg, gbeg;
	uint lt = 0, gt = 0, lttmp, gttmp, i, lfrom, gfrom, lpivot, gpivot, tmp;

	// Get the sequence block assigned to this work group
	block_record block = blocks[blockid];
	uint start = block.start, end = block.end, pivot = block.pivot, direction = block.direction;

	global parent_record* pparent = parents + block.parent; 
	global uint* psstart, *psend, *poldstart, *poldend, *pblockcount;
	global uint *s, *sn;

	// GPU-Quicksort cannot sort in place, as the regular quicksort algorithm can.
	// It therefore needs two arrays to sort things out. We start sorting in the 
	// direction of d -> dn and then change direction after each run of gqsort_kernel.
	// Which direction we are sorting: d -> dn or dn -> d?
	if (direction == 1) {
		s = d;
		sn = dn;
	} else {
		s = dn;
		sn = d;
	}

	// Align work item accesses for coalesced reads.
	// Go through data...
	for(i = start + localid; i < end; i += GQSORT_LOCAL_WORKGROUP_SIZE) {
		tmp = s[i];
		// counting elements that are smaller ...
		if (tmp < pivot)
			lt++;
		// or larger compared to the pivot.
		if (tmp > pivot) 
			gt++;
	}

	// calculate cumulative sums
	lttmp = work_group_scan_exclusive_add(lt);
	// the following barrier is here due to bug in the driver
	// please remove when updating to a new version of OpenCL 2.0 driver
	barrier(CLK_LOCAL_MEM_FENCE);
	gttmp = work_group_scan_exclusive_add(gt);

	if (localid == (GQSORT_LOCAL_WORKGROUP_SIZE - 1)) { // last work
		ltsum = lttmp+lt;
		gtsum = gttmp+gt;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// Allocate memory in the sequence this block is a part of
	if (localid == 0) {
		// get shared variables
		psstart = &pparent->sstart;
		psend = &pparent->send;
		poldstart = &pparent->oldstart;
		poldend = &pparent->oldend;
		pblockcount = &pparent->blockcount;
		// Atomic increment allocates memory to write to.
		lbeg = atomic_add(psstart, ltsum);
		// Atomic is necessary since multiple blocks access this
		gbeg = atomic_sub(psend, gtsum) - gtsum;
	}
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	// Allocate locations for work items
	lfrom = lbeg + lttmp;
	gfrom = gbeg + gttmp;

	// go thru data again writing elements to their correct position
	for(i = start + localid; i < end; i += GQSORT_LOCAL_WORKGROUP_SIZE) {
		tmp = s[i];
		// increment counts
		if (tmp < pivot) 
			sn[lfrom++] = tmp;

		if (tmp > pivot) 
			sn[gfrom++] = tmp;
	}
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	if (localid == 0) {
		if (atomic_dec(pblockcount) == 0) {
			uint sstart = *psstart;
			uint send = *psend;
			uint oldstart = *poldstart;
			uint oldend = *poldend;

			// Store the pivot value between the new sequences
			for(i = sstart; i < send; i ++) {
				d[i] = pivot;
			}

			lpivot = sn[oldstart];
			gpivot = sn[oldend-1];
			if (oldstart < sstart) {
				lpivot = median(lpivot,sn[(oldstart+sstart)/2], sn[sstart-1]);
			} 
			if (send < oldend) {
				gpivot = median(sn[send],sn[(oldend+send)/2], gpivot);
			}

			global work_record* result1 = result + 2*blockid;
			global work_record* result2 = result1 + 1;
			
			// change the direction of the sort.
			direction ^= 1;

			work_record r1 = {oldstart, sstart, lpivot, direction};
			*result1 = r1;

			work_record r2 = {send, oldend, gpivot, direction};
			*result2 = r2;
		}
	}

	// now lets recompute and relaunch
	if (get_global_id(0) == 0) {
		uint num_workgroups = get_num_groups(0);
		queue_t q = get_default_queue();
		enqueue_kernel(q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, 
				ndrange_1D(1),
				^{ relauncher_kernel(d, dn, blocks, parents, result, work, done, done_size, MAXSEQ, num_workgroups); });
	}
}

// record to push start of the sequence, end of the sequence and direction of sorting on internal stack
typedef struct workstack_record {
	uint start;
	uint end;
	uint direction;
} workstack_record;

#define PUSH(START, END) 			if (localid == 0) { \
										++workstack_pointer; \
                                        workstack_record wr = { (START), (END), direction ^ 1 }; \
										workstack[workstack_pointer] = wr; \
									} \
									barrier(CLK_LOCAL_MEM_FENCE);


//---------------------------------------------------------------------------------------
// Kernel implements the last stage of GPU-Quicksort, when all the subsequences are small
// enough to be processed in local memory. It uses similar algorithm to gqsort_kernel to 
// move items around the pivot and then switches to bitonic sort for sequences in
// the range [1, SORT_THRESHOLD].
//
// d - input array
// dn - scratch array of the same size as the input array
// seqs - array of records to be sorted in a local memory, one sequence per work group.
//---------------------------------------------------------------------------------------
kernel void lqsort_kernel(global uint* d, global uint* dn, global work_record* seqs) 
{
	const uint blockid    = get_group_id(0);
	const uint localid    = get_local_id(0);

	// workstack: stores the start and end of the sequences, direction of sort
	// If the sequence is less that SORT_THRESHOLD, it gets sorted. 
	// It will only be pushed on the stack if it greater than the SORT_THRESHOLD. 
	// Note, that the sum of ltsum + gtsum is less than QUICKSORT_BLOCK_SIZE. 
	// The total sum of the length of records on the stack cannot exceed QUICKSORT_BLOCK_SIZE, 
	// but each individual record should be greater than SORT_THRESHOLD, so the maximum length 
	// of the stack is QUICKSORT_BLOCK_SIZE/SORT_THRESHOLD - with current parameters the length 
	// of the stack is 2 :)
	local workstack_record workstack[QUICKSORT_BLOCK_SIZE/SORT_THRESHOLD]; 
	local int workstack_pointer;

	local uint mys[QUICKSORT_BLOCK_SIZE], mysn[QUICKSORT_BLOCK_SIZE], temp[SORT_THRESHOLD];
	local uint *s, *sn;
	local uint ltsum, gtsum;
	uint i, lt, gt, tmp;
	
	work_record block = seqs[blockid];
	const uint d_offset = block.start;
	uint start = 0; 
	uint end   = block.end - d_offset;

	uint direction = 1; // which direction to sort
	// initialize workstack and workstack_pointer: push the initial sequence on the stack
	if (localid == 0) {
		workstack_pointer = 0; // beginning of the stack
		workstack_record wr = { start, end, direction };
		workstack[0] = wr;
	}
	// copy block of data to be sorted by one workgroup into local memory
	// note that indeces of local data go from 0 to end-start-1
	if (block.direction == 1) {
		for (i = localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
			mys[i] = d[i+d_offset];
		}
	} else {
		for (i = localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
			mys[i] = dn[i+d_offset];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	while (workstack_pointer >= 0) { 
		// pop up the stack
		workstack_record wr = workstack[workstack_pointer];
		start = wr.start;
		end = wr.end;
		direction = wr.direction;
		barrier(CLK_LOCAL_MEM_FENCE);
		if (localid == 0) {
			--workstack_pointer;
		}
		if (direction == 1) {
			s = mys;
			sn = mysn;
		} else {
			s = mysn;
			sn = mys;
		}
		// Set local counters to zero
		lt = gt = 0;

		// Pick a pivot
		uint pivot = s[start];
		if (start < end) {
			pivot = median(pivot, s[(start+end)/2], s[end-1]);
		}
		// Align work item accesses for coalesced reads.
		// Go through data...
		for(i = start + localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
			tmp = s[i];
			// counting elements that are smaller ...
			if (tmp < pivot)
				lt++;
			// or larger compared to the pivot.
			if (tmp > pivot) 
				gt++;
		}
		
		// calculate cumulative sums
		uint lttmp = work_group_scan_exclusive_add(lt);
		uint gttmp = work_group_scan_inclusive_add(gt);
		if (localid == LQSORT_LOCAL_WORKGROUP_SIZE-1) { // last work
			ltsum = lttmp+lt;
			gtsum = gttmp;
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		// Allocate locations for work items
		uint lfrom = start + lttmp;
		uint gfrom = end - gttmp;

		// go thru data again writing elements to their correct position
		for (i = start + localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
			tmp = s[i];
			// increment counts
			if (tmp < pivot) 
				sn[lfrom++] = tmp;
			
			if (tmp > pivot) 
				sn[gfrom++] = tmp;
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		// Store the pivot value between the new sequences
		for (i = start + ltsum + localid;i < end - gtsum; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
			d[i+d_offset] = pivot;
		}
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		// if the sequence is shorter than SORT_THRESHOLD
		// sort it using an alternative sort and place result in d
		if (ltsum <= SORT_THRESHOLD) {
			sort_threshold(sn, d+d_offset, start, start + ltsum, temp, localid);
		} else {
			PUSH(start, start + ltsum);
		}
		
		if (gtsum <= SORT_THRESHOLD) {
			sort_threshold(sn, d+d_offset, end - gtsum, end, temp, localid);
		} else {
			PUSH(end - gtsum, end);
		}
	}
}