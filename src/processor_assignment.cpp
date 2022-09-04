#include "processor_assignment.h"
#include "utils.h"

void processor_assignment::ScheduleTasksOnProcessors(Graph<edge_t>* DAG, metrics_t* ordered_metrics)
{
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

	edge_t* predecessors = DAG->GetWeightsArray();
	edge_t* edges = DAG->GetEdgesArray();

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
					weight_for_max_aft_predecessor = DAG->GetWeightsArray()[matrix_to_array_indexes(j, current_node, DAG->len)];
				}
			}
		}

		int eft_min = INT_MAX;
		for (int processor = 0; processor < DAG->number_of_processors; processor++) {
			int cost_on_processor = DAG->GetCostsArray()[matrix_to_array_indexes(current_node, processor, DAG->number_of_processors)];
			int cost_of_predecessor_in_same_processor = 0;
			int cost_of_predecessors_in_different_processors = 0;
			if (processor_for_max_aft_predecessor == processor) {
				cost_of_predecessor_in_same_processor = weight_for_max_aft_predecessor; //se max eft è nello stesso processo, tolgo il costo di trasferimento.
			}
			
			//controlla cosa succede al task 11
			//a causa del fatto che il peso del predecessore nello stesso processore va tolto, 
			//mentre il massimo di quelli in altri processori va aggiunto, 
			//ma solo se il costo più l'eft dei task negli altri processori supera l'eft meno il costo di quello nello stesso processore, 
			//e in quel caso devo aggiungere la differenza.
			//inoltre dipende se l'eft max è di un task nello stesso processore o meno.
			//inoltre se più di un task predecessori sono sullo stesso processore devo togliere il peso ad entrambi e vedere cosa succede.

			int remaining_transfer_cost = 0;
			for (int j = 0; j < DAG->max_parents_for_nodes; j++)
			{
				int currentParent = edges[matrix_to_array_indexes(j, current_node, DAG->len)];
				if (currentParent > -1 && currentParent != predecessor_with_max_aft) {
					cost_of_predecessors_in_different_processors = max(
						cost_of_predecessors_in_different_processors, 
						task_processor_assignment[currentParent].z + predecessors[matrix_to_array_indexes(j, current_node, DAG->len)]);
				}

				remaining_transfer_cost = max(max_aft_of_predecessors - cost_of_predecessor_in_same_processor, cost_of_predecessors_in_different_processors);
			}

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
		print(task_processor_assignment, n_nodes, "\n", true, 0);
	}

	cout << "processors assigned: (processor id, EST, EFT)" << endl;
	if (DEBUG_PROCESSOR_ASSIGNMENT) {
		print(task_processor_assignment, n_nodes, "\n", true, 0);
	}
	else { //mostro solo i primi e gli ultimi 5 elementi per essere sicuro che tutto abbia funzionato.
		print(task_processor_assignment, min(n_nodes, 5), "\n", true, 0);
		cout << "[...]" << endl << endl;
		print(task_processor_assignment, n_nodes, "\n", true, n_nodes - 5);
	}
	cout << endl;

	delete[] processorsNextSlotStart;
	delete[] task_processor_assignment;
}
