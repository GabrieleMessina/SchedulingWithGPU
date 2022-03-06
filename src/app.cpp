#include "app_globals.h"
#if DEBUG_MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <crtdbg.h>
#endif // DEBUG_MEMORY_LEAK
#include <cstdlib>
#include "utils.h"
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <sstream>
#include <chrono>
#include <ctime>
#include <tuple>
#include "CL/cl.h"
#include "ocl_manager.h"
#include "dag.h"
#include "ocl_buffer_manager.h"
#include "entry_discover.h"
#include "compute_metrics.h"
#include "sort_metrics.h"

using namespace std;

Graph<int> *DAG;
int *entrypoints;
cl_int2 *metrics, *ordered_metrics;

void printMemoryUsage();
void measurePerformance(cl_event entry_discover_evt,cl_event *compute_metrics_evt, cl_event *sort_task_evts, int nels);
void verify();

std::chrono::system_clock::time_point start_time;
std::chrono::system_clock::time_point end_time;

int repeatNTimes = 1;
bool isVectorizedVersion = false;
string dataSetName;
string userResponseToVectorizeQuestion;

int main(int argc, char *argv[]) {
start:
	if (argc <= 1) {
		//usage("syntax: graph_init datasetName [vector4 version (false)] [how many times? (1)]");
		cout << "Nome del dataset: ";
		cin >> dataSetName;
		cout << "Vettorizzare? true/false: ";
		cin >> userResponseToVectorizeQuestion;
		cout << "Quante volte ripetere? ";
		cin >> repeatNTimes;
	} else {
		dataSetName = argv[1];
		if (argc <= 3) {
			userResponseToVectorizeQuestion = argv[2];
			cout << "Quante volte ripetere? ";
			cin >> repeatNTimes;
		}
		if (argc > 3) {
			repeatNTimes = atoi(argv[3]);
		}
	}
	isVectorizedVersion = (strcmp(userResponseToVectorizeQuestion.c_str(), "true") == 0);
	isVectorizedVersion |= (strcmp(userResponseToVectorizeQuestion.c_str(), "s") == 0);
	isVectorizedVersion |= (strcmp(userResponseToVectorizeQuestion.c_str(), "1") == 0);

	if(isVectorizedVersion) OCLManager::InitVectorized(VectorizedComputeMetricsVersion::RectangularV2); //TODO: make the user choose the version
	else OCLManager::Init(ComputeMetricsVersion::Rectangular);

	//LEGGERE IL DATASET E INIZIALIZZARE LA DAG
	DAG = Graph<int>::initDagWithDataSet(dataSetName.c_str());
	int n_nodes = DAG->len;

	for (int i = 0; i < repeatNTimes; i++)
	{

		start_time = std::chrono::system_clock::now();
		if(isVectorizedVersion)
			cout<<"vectorized version"<<endl;
		else cout<<"standard version"<<endl;

		OCLBufferManager::Init(n_nodes, DAG->adj_len, isVectorizedVersion);


		cl_event entry_discover_evt;
		std::tie(entry_discover_evt, entrypoints) = EntryDiscover::Run(DAG);


		cl_event *compute_metrics_evt;
		if(isVectorizedVersion)
			std::tie(compute_metrics_evt, metrics) = ComputeMetrics::RunVectorized(DAG, entrypoints);
		else 
			std::tie(compute_metrics_evt, metrics) = ComputeMetrics::Run(DAG, entrypoints);

		cl_event* sort_task_evts;
		std::tie(sort_task_evts, ordered_metrics) = SortMetrics::MergeSort(metrics, n_nodes);

		end_time = std::chrono::system_clock::now();

		//METRICHE
		measurePerformance(entry_discover_evt, compute_metrics_evt, sort_task_evts, n_nodes);
		////VERIFICA DELLA CORRETTEZZA
		verify();

		//PULIZIA FINALE
		delete[] entrypoints;
		delete[] metrics;
		delete[] ordered_metrics;

		delete[] compute_metrics_evt;
		delete[] sort_task_evts;

		OCLBufferManager::Release();
		OCLManager::Reset();
		cout << "-----------------END LOOP " << i+1 << "/" << repeatNTimes << "---------------------" << endl;
	}
	delete DAG;

	cout << "-----------------------------------------" << endl;
	cout << "-----------------END---------------------" << endl;
	cout << "-----------------------------------------" << endl << endl;

#if DEBUG_MEMORY_LEAK
	_CrtDumpMemoryLeaks();
#endif // DEBUG_MEMORY_LEAK

	system("PAUSE");

#if WINDOWS
	exec("cls");
#else
	exec("clear");
#endif

	goto start;
}

void measurePerformance(cl_event entry_discover_evt,cl_event *compute_metrics_evt, cl_event *sort_task_evts, int nels) {
	double runtime_discover_ms = runtime_ms(entry_discover_evt);
	double runtime_metrics_ms = total_runtime_ms(compute_metrics_evt[0], compute_metrics_evt[1]);
	double gap_discover_metrics = total_runtime_ms(entry_discover_evt, compute_metrics_evt[0]);
	double runtime_sorts_ms =  total_runtime_ms(sort_task_evts[0], sort_task_evts[1]);
	double gap_metrics_sort = total_runtime_ms(compute_metrics_evt[1], sort_task_evts[0]);

	std::time_t end_time_t = std::chrono::system_clock::to_time_t(end_time);
	std::chrono::duration<double, std::milli> elapsed_seconds = end_time-start_time;
	tm *end_date_time_info = localtime(&end_time_t);
	char end_date_time[80];
	strftime(end_date_time, 80, "%d/%m/%y %H.%M.%S", end_date_time_info);
	end_date_time[strcspn(end_date_time , "\n")] = 0; //remove new line

	double total_elapsed_time_GPU = total_runtime_ms(entry_discover_evt, sort_task_evts[1]);
	int platform_id = 0;
	char *m_platform_name = getSelectedPlatformInfo(platform_id);
	int device_id = 0;
	char *m_device_name = getSelectedDeviceInfo(device_id);

	int gpu_temperature = -1;
	int cpu_temperature = -1;
#if WINDOWS
	string gpu_temperature_string = exec(".\\utils\\get_current_gpu_temperature.cmd");
	if(gpu_temperature_string.length() != 0) {
		size_t start_sub = gpu_temperature_string.find(':')+1, end_sub = gpu_temperature_string.find_last_of('C')-1;
		gpu_temperature_string = gpu_temperature_string.substr(start_sub, end_sub-start_sub);
		gpu_temperature = atoi(gpu_temperature_string.c_str());
	}
	string cpu_temperature_string = exec(".\\utils\\get_current_cpu_temperature.cmd");
	if(cpu_temperature_string.length() != 0) {
		cpu_temperature = atoi(cpu_temperature_string.c_str());
	}
#endif // WINDOWS

	//stampo i file in un .csv per poter analizzare i dati successivamente.
	FILE *fp;
	fp = fopen("results/execution_results.csv", "a");
	if (fp == NULL) {
		printf("Error opening result file!\n");
		exit(1);
	}
	if(is_file_empty(fp))
		fprintf(fp, "DATA; N TASKS; TOTAL RUN SECONDS CPU; TOTAL RUN SECONDS GPU; DISCOVER ENTRIES RUNTIME MS; COMPUTE METRICS RUNTIME MS; SORT RUNTIME MS; VERSION; PLATFORM; PREFERRED WORK GROUP SIZE; GPU TEMPERATURE; CPU TEMPERATURE; DEVICE\n");

	fprintf(fp, "%s; %d; %E; %E; %E; %E; %E; %s; %s; %d; %d; %d; %s\n",
		end_date_time, 
		nels,
		elapsed_seconds.count(),
		total_elapsed_time_GPU,
		runtime_discover_ms,
		runtime_metrics_ms,
		runtime_sorts_ms,
		(isVectorizedVersion ? "vectorized" : "standard"),
		m_platform_name,
		OCLManager::preferred_wg_size,
		gpu_temperature,
		cpu_temperature,
		m_device_name
	);

	if (DEBUG_METRICS) {
		printf("%s; %d; %E; %E; %E; %E; %E; %E; %E; %s; %s; %d; %d; %d; %s\n",
			end_date_time,
			nels,
			elapsed_seconds.count(),
			total_elapsed_time_GPU,
			runtime_discover_ms,
			runtime_metrics_ms,
			runtime_sorts_ms,
			gap_discover_metrics,
			gap_metrics_sort,
			(isVectorizedVersion ? "vectorized" : "standard"),
			m_platform_name,
			OCLManager::preferred_wg_size,
			gpu_temperature,
			cpu_temperature,
			m_device_name
		);
		printMemoryUsage();
	}
	//somma di byte letti e scritti dal kernel diviso tempo di esecuzione
	//if (DEBUG_METRICS) {
	//	//TODO: check the math as algorithms changed
	//	printf("discover entries: runtime %.4gms, %.4g GE/s, %.4g GB/s\n",
	//		runtime_discover_ms, nels / runtime_discover_ms / 1.0e6, OCLManager::preferred_wg_size / runtime_discover_ms / 1.0e6);
	//	printf("compute metrics: runtime %.4gms, %.4g GE/s, %.4g GB/s\n",
	//		runtime_metrics_ms, nels / runtime_metrics_ms / 1.0e6, OCLManager::preferred_wg_size / runtime_metrics_ms / 1.0e6);
	//	printf("sort tasks: runtime %.4gms, %.4g GE/s, %.4g GB/s\n",
	//		runtime_sorts_ms, nels / runtime_sorts_ms / 1.0e6, OCLManager::preferred_wg_size / runtime_sorts_ms / 1.0e6);
	//	printMemoryUsage();
	//}

	fflush(fp);
	fclose(fp);
}

void verify() {
	//TODO: calcolare le metriche su CPU e verificare che siano identiche a quelle calcolate su GPU;
	//scandire ordered_metrics e verificare che sia ordinati
	for (int i = 0; i < DAG->len - 1; ++i) {
		if(ordered_metrics[i] > ordered_metrics[i+1]) {
			fprintf(stderr, "ordered_metrics[%d] = (%d, %d) > ordered_metrics[%d] = (%d, %d)\n", i, ordered_metrics[i].x, ordered_metrics[i].y, i+1, ordered_metrics[i+1].x, ordered_metrics[i+1].y);
			error("mismatch");
		}
	}
	printf("Everything sorted, verified\n");
}
