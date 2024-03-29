#include "app_globals.h"
#if DEBUG_MEMORY_LEAK
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#endif // DEBUG_MEMORY_LEAK
#include <cstdlib>
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
#include <filesystem>
namespace fs = std::filesystem;
#include "CL/cl.h"
#include "ocl_manager.h"
#include "dag.h"
#include "ocl_buffer_manager.h"
#include "entry_discover.h"
#include "compute_metrics.h"
#include "sort_metrics.h"
#include "processor_assignment.h"
#include "utils.h"

using namespace std;

Graph<int>* DAG;
int* entrypoints;
metrics_t* metrics, * ordered_metrics;

void printMemoryUsage();
void measurePerformance(cl_event entry_discover_evt, cl_event* compute_metrics_evt, cl_event* sort_task_evts, cl_event* processor_assignment_evts, int nels, int nProc);
void verify();

std::chrono::system_clock::time_point start_time;
std::chrono::system_clock::time_point middle_time1;
std::chrono::system_clock::time_point middle_time2;
std::chrono::system_clock::time_point middle_time3;
std::chrono::system_clock::time_point middle_time4;
std::chrono::system_clock::time_point middle_time5;
std::chrono::system_clock::time_point middle_time6;
std::chrono::system_clock::time_point middle_time7;
std::chrono::system_clock::time_point middle_time8;
std::chrono::system_clock::time_point end_time;

int repeatNTimes = 20;
bool isVectorizedVersion = true;
string dataSetName = "";
string userResponseToVectorizeQuestion = "1";

int main(int argc, char* argv[]) {
	try
	{
	start:
		//if (argc <= 1) {
		//	//usage("syntax: graph_init datasetName [vector4 version (false)] [how many times? (1)]");
		//	cout << "Nome del dataset: ";
		//	cin >> dataSetName;
		//	cout << "Vettorizzare? true/false: ";
		//	cin >> userResponseToVectorizeQuestion;
		//	cout << "Quante volte ripetere? ";
		//	cin >> repeatNTimes;
		//}
		//else {
		//	dataSetName = argv[1];
		//	if (argc <= 3) {
		//		userResponseToVectorizeQuestion = argv[2];
		//		cout << "Quante volte ripetere? ";
		//		cin >> repeatNTimes;
		//	}
		//	if (argc > 3) {
		//		repeatNTimes = atoi(argv[3]);
		//	}
		//}
		/*isVectorizedVersion = (strcmp(userResponseToVectorizeQuestion.c_str(), "true") == 0);
		isVectorizedVersion |= (strcmp(userResponseToVectorizeQuestion.c_str(), "s") == 0);
		isVectorizedVersion |= (strcmp(userResponseToVectorizeQuestion.c_str(), "1") == 0);*/

		if (isVectorizedVersion) OCLManager::InitVectorized(VectorizedComputeMetricsVersion::RectangularVec8);
		else OCLManager::Init(ComputeMetricsVersion::Rectangular);

		std::string path = ".\\data_set\\raw";
		for (const auto& entry : fs::directory_iterator(path)) {
			dataSetName = entry.path().string();

			//LEGGERE IL DATASET E INIZIALIZZARE LA DAG
			DAG = Graph<int>::initDagWithDataSet(dataSetName.c_str());
			int n_nodes = DAG->len;

			/*if (isVectorizedVersion)cout << "vectorized version" << endl << endl;
			else cout << "standard version" << endl << endl;*/
			for (int i = 0; i < repeatNTimes; i++)
			{
				//int count = 0;
				//cout << count++ << endl;
				start_time = std::chrono::system_clock::now();

				OCLBufferManager* bufferMananger = OCLBufferManager::Init(n_nodes, DAG->adj_len, DAG->adj_reverse_len, DAG->number_of_processors, isVectorizedVersion);

				//middle_time1 = std::chrono::system_clock::now();
				cl_event entry_discover_evt;
				std::tie(entry_discover_evt, entrypoints) = EntryDiscover::Run(DAG);
				//middle_time2 = std::chrono::system_clock::now();

				//middle_time3 = std::chrono::system_clock::now();
				cl_event* compute_metrics_evt;
				if (isVectorizedVersion) std::tie(compute_metrics_evt, metrics) = ComputeMetrics::RunVectorized(DAG, entrypoints);
				else std::tie(compute_metrics_evt, metrics) = ComputeMetrics::Run(DAG, entrypoints);
				//middle_time4 = std::chrono::system_clock::now();

				
				//middle_time5 = std::chrono::system_clock::now();
				cl_event* sort_task_evts;
				if (n_nodes < OCLManager::preferred_wg_size * 2) {
					std::tie(sort_task_evts, ordered_metrics) = SortMetrics::MergeSort(metrics, n_nodes);
				}
				else {
					std::tie(sort_task_evts, ordered_metrics) = SortMetrics::BitonicMergesort(metrics, n_nodes);
				}
				//middle_time6 = std::chrono::system_clock::now();

				middle_time7 = std::chrono::system_clock::now();
				cl_event* processor_assignment_evts;
				processor_assignment_evts = processor_assignment::ScheduleTasksOnProcessors(DAG, ordered_metrics);
				middle_time8 = std::chrono::system_clock::now();

				end_time = std::chrono::system_clock::now();

				//METRICHE
				measurePerformance(entry_discover_evt, compute_metrics_evt, sort_task_evts, /*processor_assignment_evts*/ /*(non usiamo la GPU)*/ sort_task_evts, n_nodes, DAG->number_of_processors);

				//PULIZIA FINALE
				delete[] entrypoints;
				delete[] metrics;
				delete[] ordered_metrics;

				delete[] compute_metrics_evt;
				delete[] sort_task_evts;
				delete[] processor_assignment_evts;

				bufferMananger->Release();
				OCLManager::Reset();
				//cout << "-----------------END LOOP " << i + 1 << "/" << repeatNTimes << "---------------------" << endl;
			}
			delete DAG;
		}
		OCLManager::Release();

	/*	cout << "-----------------------------------------" << endl;
		cout << "-----------------END---------------------" << endl;
		cout << "-----------------------------------------" << endl << endl;*/

#if DEBUG_MEMORY_LEAK
		_CrtDumpMemoryLeaks();
#endif // DEBUG_MEMORY_LEAK

		system("PAUSE");
//
//#if WINDOWS
//		exec("cls");
//#else
//		exec("clear");
//#endif

		//goto start;
	}
	catch (const std::exception& e)
	{
		cerr << e.what() << endl;
	}
	return 0;
}

void measurePerformance(cl_event entry_discover_evt, cl_event* compute_metrics_evt, cl_event* sort_task_evts, cl_event* processor_assignment_evts, int nels, int nProc) {
	double runtime_discover_ms = runtime_ms(entry_discover_evt);
	double runtime_metrics_ms = total_runtime_ms(compute_metrics_evt[0], compute_metrics_evt[1]);
	double gap_discover_metrics = total_runtime_ms(entry_discover_evt, compute_metrics_evt[0]);
	double runtime_sorts_ms = total_runtime_ms(sort_task_evts[0], sort_task_evts[1]);
	double gap_metrics_sort = total_runtime_ms(compute_metrics_evt[1], sort_task_evts[0]);
	double runtime_proc_ass_ms = total_runtime_ms(processor_assignment_evts[0], processor_assignment_evts[1]);

	std::time_t end_time_t = std::chrono::system_clock::to_time_t(end_time);
	std::chrono::duration<double, std::milli> elapsed_seconds = end_time - start_time;
	/*std::chrono::duration<double, std::milli> elapsed_seconds1 = middle_time2 - middle_time1;
	std::chrono::duration<double, std::milli> elapsed_seconds2 = middle_time4 - middle_time3;
	std::chrono::duration<double, std::milli> elapsed_seconds3 = middle_time6 - middle_time5;*/
	std::chrono::duration<double, std::milli> elapsed_seconds4 = middle_time8 - middle_time7;
	tm* end_date_time_info = localtime(&end_time_t);
	char end_date_time[80];
	strftime(end_date_time, 80, "%d/%m/%y %H.%M.%S", end_date_time_info);
	end_date_time[strcspn(end_date_time, "\n")] = 0; //remove new line

	/*printf("Entrypoint CPU: %E, GPU: %E\n", elapsed_seconds1.count(), runtime_discover_ms);
	printf("Compute Me CPU: %E, GPU: %E\n", elapsed_seconds2.count(), runtime_metrics_ms);
	printf("Sort metri CPU: %E, GPU: %E\n", elapsed_seconds3.count(), runtime_sorts_ms);
	printf("Processor  CPU: %E, GPU: N/A\n", elapsed_seconds4.count());*/
	runtime_proc_ass_ms = elapsed_seconds4.count();

	double total_elapsed_time_GPU = total_runtime_ms(entry_discover_evt, processor_assignment_evts[1]);
	int platform_id = 0;
	char* m_platform_name = getSelectedPlatformInfo(platform_id);
	int device_id = 0;
	char* m_device_name = getSelectedDeviceInfo(device_id);

	int gpu_temperature = -1;
	int cpu_temperature = -1;
#if WINDOWS
	string gpu_temperature_string = exec(".\\utils\\get_current_gpu_temperature.cmd");
	if (gpu_temperature_string.length() != 0) {
		size_t start_sub = gpu_temperature_string.find(':') + 1, end_sub = gpu_temperature_string.find_last_of('C') - 1;
		gpu_temperature_string = gpu_temperature_string.substr(start_sub, end_sub - start_sub);
		gpu_temperature = atoi(gpu_temperature_string.c_str());
	}
	/*string cpu_temperature_string = exec(".\\utils\\get_current_cpu_temperature.cmd");
	if (cpu_temperature_string.length() != 0) {
		cpu_temperature = atoi(cpu_temperature_string.c_str());
	}*/
#endif // WINDOWS

	//stampo i file in un .csv per poter analizzare i dati successivamente.
	FILE* fp;
	fp = fopen("results/execution_results.csv", "a");
	if (fp == NULL) {
		printf("Error opening result file!\n");
		exit(1);
	}
	if (is_file_empty(fp))
		fprintf(fp, "DATA; DATASET_NAME; N TASKS; N PROC; TOTAL RUN SECONDS CPU; TOTAL RUN SECONDS GPU; DISCOVER ENTRIES RUNTIME MS; COMPUTE METRICS RUNTIME MS; SORT RUNTIME MS; PROC ASSIGN RUNTIME MS; VERSION; PLATFORM; PREFERRED WORK GROUP SIZE; GPU TEMPERATURE; CPU TEMPERATURE; DEVICE\n");

	fprintf(fp, "%s; %s; %d; %d; %E; %E; %E; %E; %E; %E; %s; %s; %d; %d; %d; %s\n",
		end_date_time,
		dataSetName.c_str(),
		nels,
		nProc,
		elapsed_seconds.count(),
		total_elapsed_time_GPU,
		runtime_discover_ms,
		runtime_metrics_ms,
		runtime_sorts_ms,
		runtime_proc_ass_ms,
		(isVectorizedVersion ? "vectorized" : "standard"),
		m_platform_name,
		OCLManager::preferred_wg_size,
		gpu_temperature,
		cpu_temperature,
		m_device_name
	);

	if (DEBUG_OCL_METRICS) {
		printf("%s; %s; %d; %d; %E; %E; %E; %E; %E; %E; %E; %E; %s; %s; %d; %d; %d; %s\n",
			end_date_time,
			dataSetName.c_str(),
			nels,
			nProc,
			elapsed_seconds.count(),
			total_elapsed_time_GPU,
			runtime_discover_ms,
			runtime_metrics_ms,
			runtime_sorts_ms,
			gap_discover_metrics,
			gap_metrics_sort,
			runtime_proc_ass_ms,
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
	//if (DEBUG_OCL_METRICS) {
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
	//TODO: calcolare il risultato finale e verificare che sia identico a quelle calcolate su GPU;
}
