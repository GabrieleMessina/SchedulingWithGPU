#include <chrono>
#include <ctime>
#include "processor_assignment.h"
#include "ocl_manager.h"
#include "ocl_buffer_manager.h"
#include "utils.h"
cl_event* processor_assignment::ScheduleTasksOnProcessors(Graph<edge_t>* DAG, metrics_t* ordered_metrics)
{
	OCLBufferManager BufferManager = *OCLBufferManager::GetInstance();
	cl_event *startEvent = NULL, *endEvent = NULL;
	// per ogni livello calcola EST e EFT di ogni task su ogni processore

	//AFT = actual finish time
	//EFT = Earliest finish time
	//EST = Earliest start time
	//est(v,p) = max(nextPossibleSlotInProcessorStart, max_per_ogni_pred(AFT(pred di v) + weights(pred, v)));
	//eft(v,p) = est(v,p) + costOnProcessor p di v;

	//nel paper il DTC viene indicato come valore somma dei singoli pesi degli edge, ma in maniera abbastanza nascosta
	// quando viene usato nei calcoli viene diviso per il numero di successori di un task o comunque viene usato il costo singolo dell'arco.

	//per ogni livello si estraggono dalla coda creata sui rank i task, e si assegnato man mano al loro processore preferito,
	//cioè quello che fornisce EFT minore.

	int n_nodes = DAG->len;
	const int metrics_len = GetMetricsArrayLenght(n_nodes);
	int* processorsNextSlotStart = new int[DAG->number_of_processors];
	for (int i = 0; i < DAG->number_of_processors; i++) processorsNextSlotStart[i] = 0;
	cl_int3* task_processor_assignment = new cl_int3[n_nodes]; //per ogni task, processorId, EST, EFT
	for (int i = 0; i < n_nodes; i++) task_processor_assignment[i] = cl_int3{ -1,-1,-1 };
	cl_int3* output = DBG_NEW cl_int3[DAG->number_of_processors]; 
	for (int i = 0; i < DAG->number_of_processors; i++) output[i] = cl_int3{ 0,0,0 };


	edge_t* predecessors = DAG->GetWeightsArray();
	edge_t* edges = DAG->GetEdgesArray();
	edge_t* costs = DAG->GetCostsArray();


	for (int i = 0; i < metrics_len; i++)//per ogni task in ordine di metrica
	{
		int current_node = ordered_metrics[i].z; //index in dag del task
		if (current_node >= n_nodes) continue; //probabilmente task di padding.
		int predecessor_with_max_aft = -1;
		int max_aft_of_predecessors = -1;
		int processor_for_max_aft_predecessor = -1;
		int weight_for_max_aft_predecessor = 0;

		//find the predecessor with the max EFT
		//Ciclo sui suoi successori e controllo in task_processor_assignment
		for (int j = 0; j < DAG->max_parents_for_nodes; j++)
		{
			int currentParent = edges[matrix_to_array_indexes(j, current_node, DAG->len)];
			if (currentParent > -1) {
				int edge_weight_with_parent = predecessors[matrix_to_array_indexes(j, current_node, DAG->len)];
				int parentEFT = task_processor_assignment[currentParent].z + edge_weight_with_parent;
				if (parentEFT > max_aft_of_predecessors) {
					max_aft_of_predecessors = parentEFT;
					predecessor_with_max_aft = currentParent;
					processor_for_max_aft_predecessor = task_processor_assignment[currentParent].x;
					weight_for_max_aft_predecessor = edge_weight_with_parent;
				}
			}
		}



		int eft_min = INT_MAX;

		int cost_of_predecessors_in_different_processors = 0;
		int remaining_transfer_cost = 0;
		for (int j = 0; j < DAG->max_parents_for_nodes; j++)
		{
			int currentParent = edges[matrix_to_array_indexes(j, current_node, DAG->len)];
			if (currentParent > -1 && currentParent != predecessor_with_max_aft) {
				cost_of_predecessors_in_different_processors = max(
					cost_of_predecessors_in_different_processors,
					task_processor_assignment[currentParent].z + predecessors[matrix_to_array_indexes(j, current_node, DAG->len)]);
			}
		}
		
		
		//cl_event processor_cost_evt = run_kernel(DAG, current_node, task_processor_assignment, processorsNextSlotStart, predecessor_with_max_aft,
		//	weight_for_max_aft_predecessor, max_aft_of_predecessors, costs, processor_for_max_aft_predecessor, cost_of_predecessors_in_different_processors);
		//
		//BufferManager.GetProcessorsCostResult(output, &processor_cost_evt, 1);
		//
		//if (startEvent == NULL) startEvent = &processor_cost_evt;
		//endEvent = &processor_cost_evt;

		////print(output, DAG->number_of_processors, "\n", true, 0);

		//for (int i = 0; i < DAG->number_of_processors; i++) {
		//	int eft = output[i].y;
		//	if (eft < eft_min) {
		//		eft_min = eft;
		//		task_processor_assignment[current_node] = output[i];
		//	}
		//}

		for (int processor = 0; processor < DAG->number_of_processors; processor++) {
			int cost_of_predecessor_in_same_processor = 0;
			int cost_on_processor = costs[matrix_to_array_indexes(current_node, processor, DAG->number_of_processors)];
			if (processor_for_max_aft_predecessor == processor) {
				cost_of_predecessor_in_same_processor = weight_for_max_aft_predecessor; //se max eft è nello stesso processo, tolgo il costo di trasferimento.
			}
			remaining_transfer_cost = max(max_aft_of_predecessors - cost_of_predecessor_in_same_processor, cost_of_predecessors_in_different_processors);
			
			//controlla cosa succede al task 11
			//a causa del fatto che il peso del predecessore nello stesso processore va tolto, 
			//mentre il massimo di quelli in altri processori va aggiunto, 
			//ma solo se il costo più l'eft dei task negli altri processori supera l'eft meno il costo di quello nello stesso processore, 
			//e in quel caso devo aggiungere la differenza.
			//inoltre dipende se l'eft max è di un task nello stesso processore o meno.
			//inoltre se più di un task predecessori sono sullo stesso processore devo togliere il peso ad entrambi e vedere cosa succede.

			//finalmente posso aggiungere l'eft al task.
			int est = max(processorsNextSlotStart[processor], /*max_aft_of_predecessors +*/ remaining_transfer_cost);
			int eft = est + cost_on_processor;
			if (eft < eft_min) {
				eft_min = eft;
				task_processor_assignment[current_node] = cl_int3{ processor, est, eft };
			}
		}

		//assegno il tempo di occupazione al processore sulla base del tempo previsto per il task.
		processorsNextSlotStart[task_processor_assignment[current_node].x] = task_processor_assignment[current_node].z;
	}

	if (DEBUG_PROCESSOR_ASSIGNMENT) {
		cout << "processors assigned: (processor id, EST, EFT)" << endl;
		print(task_processor_assignment, n_nodes, "\n", true, 0);
	}
	else if(DEBUG_PROCESSOR_ASSIGNMENT_PARTIAL){ //mostro solo i primi e gli ultimi 5 elementi per essere sicuro che tutto abbia funzionato.
		cout << "processors assigned: (processor id, EST, EFT)" << endl;
		print(task_processor_assignment, min(n_nodes, 5), "\n", true, 0);
		cout << "[...]" << endl << endl;
		print(task_processor_assignment, n_nodes, "\n", true, n_nodes - 5);
	}
	cout << endl;

	BufferManager.ReleaseGraphEdges();
	BufferManager.ReleaseGraphWeightsReverse();
	BufferManager.ReleaseProcessorsCost();
	BufferManager.ReleaseTaskProcessorAssignment();
	BufferManager.ReleaseProcessorNextSlotStart();

	delete[] processorsNextSlotStart;
	delete[] task_processor_assignment;
	delete[] output;

	cl_event* events = DBG_NEW cl_event[2];
	//events[0] = *startEvent;
	//events[1] = *endEvent;

	return events;
}



cl_event processor_assignment::run_kernel(Graph<edge_t>* DAG, int current_node, cl_int3* task_processor_assignment, int* processorsNextSlotStart,
	int predecessor_with_max_aft, int weight_for_max_aft_predecessor, int max_aft_of_predecessors, edge_t* costs, 
	int processor_for_max_aft_predecessor, int cost_of_predecessors_in_different_processors) {
	OCLBufferManager BufferManager = *OCLBufferManager::GetInstance();

	int arg_index = 0;
	const size_t lws[] = { OCLManager::preferred_wg_size };
	const size_t gws[] = { round_mul_up(DAG->number_of_processors, lws[0]) };

	cl_int err;

	BufferManager.SetTaskProcessorAssignment(task_processor_assignment);
	BufferManager.SetProcessorNextSlotStart(processorsNextSlotStart);

	cl_mem edges_GPU = BufferManager.GetGraphEdges();
	cl_mem predecessors_GPU = BufferManager.GetGraphWeightsReverse();
	cl_mem output_GPU = BufferManager.GetProcessorsCost();
	cl_mem task_processor_assignment_GPU = BufferManager.GetTaskProcessorAssignment();
	cl_mem processorsNextSlotStart_GPU = BufferManager.GetProcessorNextSlotStart();
	cl_mem costs_GPU = BufferManager.GetCostsOnProcessor();
	int max_edges_dept = DAG->max_parents_for_nodes;
	int len = DAG->len;
	int number_of_processors = DAG->number_of_processors;

	err = clSetKernelArg(OCLManager::GetComputeProcessorCostKernel(), arg_index++, sizeof(current_node), &current_node);
	ocl_check(err, "set arg %d for compute_processor_cost_k", arg_index);
	err = clSetKernelArg(OCLManager::GetComputeProcessorCostKernel(), arg_index++, sizeof(edges_GPU), &edges_GPU);
	ocl_check(err, "set arg %d for compute_processor_cost_k", arg_index);
	err = clSetKernelArg(OCLManager::GetComputeProcessorCostKernel(), arg_index++, sizeof(predecessors_GPU), &predecessors_GPU);
	ocl_check(err, "set arg %d for compute_processor_cost_k", arg_index);
	err = clSetKernelArg(OCLManager::GetComputeProcessorCostKernel(), arg_index++, sizeof(costs_GPU), &costs_GPU);
	ocl_check(err, "set arg %d for compute_processor_cost_k", arg_index);
	err = clSetKernelArg(OCLManager::GetComputeProcessorCostKernel(), arg_index++, sizeof(max_edges_dept), &max_edges_dept);
	ocl_check(err, "set arg %d for compute_processor_cost_k", arg_index);	
	err = clSetKernelArg(OCLManager::GetComputeProcessorCostKernel(), arg_index++, sizeof(output_GPU), &output_GPU);
	ocl_check(err, "set arg %d for compute_processor_cost_k", arg_index);
	err = clSetKernelArg(OCLManager::GetComputeProcessorCostKernel(), arg_index++, sizeof(task_processor_assignment_GPU), &task_processor_assignment_GPU);
	ocl_check(err, "set arg %d for compute_processor_cost_k", arg_index);
	err = clSetKernelArg(OCLManager::GetComputeProcessorCostKernel(), arg_index++, sizeof(processorsNextSlotStart_GPU), &processorsNextSlotStart_GPU);
	ocl_check(err, "set arg %d for compute_processor_cost_k", arg_index);
	err = clSetKernelArg(OCLManager::GetComputeProcessorCostKernel(), arg_index++, sizeof(len), &len);
	ocl_check(err, "set arg %d for compute_processor_cost_k", arg_index);
	err = clSetKernelArg(OCLManager::GetComputeProcessorCostKernel(), arg_index++, sizeof(number_of_processors), &number_of_processors);
	ocl_check(err, "set arg %d for compute_processor_cost_k", arg_index);
	err = clSetKernelArg(OCLManager::GetComputeProcessorCostKernel(), arg_index++, sizeof(predecessor_with_max_aft), &predecessor_with_max_aft);
	ocl_check(err, "set arg %d for compute_processor_cost_k", arg_index);
	err = clSetKernelArg(OCLManager::GetComputeProcessorCostKernel(), arg_index++, sizeof(weight_for_max_aft_predecessor), &weight_for_max_aft_predecessor);
	ocl_check(err, "set arg %d for compute_processor_cost_k", arg_index); 
	err = clSetKernelArg(OCLManager::GetComputeProcessorCostKernel(), arg_index++, sizeof(processor_for_max_aft_predecessor), &processor_for_max_aft_predecessor);
	ocl_check(err, "set arg %d for compute_processor_cost_k", arg_index); 
	err = clSetKernelArg(OCLManager::GetComputeProcessorCostKernel(), arg_index++, sizeof(cost_of_predecessors_in_different_processors), &cost_of_predecessors_in_different_processors);
	ocl_check(err, "set arg %d for compute_processor_cost_k", arg_index); 
	err = clSetKernelArg(OCLManager::GetComputeProcessorCostKernel(), arg_index++, sizeof(max_aft_of_predecessors), &max_aft_of_predecessors);
	ocl_check(err, "set arg %d for compute_processor_cost_k", arg_index);


	cl_event compute_processor_cost_evt;
	err = clEnqueueNDRangeKernel(OCLManager::queue,
		OCLManager::GetComputeProcessorCostKernel(),
		1, NULL, gws, lws,
		0, NULL, &compute_processor_cost_evt);

	ocl_check(err, "enqueue compute_processor_cost kernel");

	return compute_processor_cost_evt;
}
