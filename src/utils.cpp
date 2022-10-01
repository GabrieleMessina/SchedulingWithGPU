#include <math.h>
#include "app_globals.h"
#if WINDOWS
#include "windows.h"
#include "psapi.h"
#endif
#include "utils.h"

void error(char const* str) {
	fprintf(stderr, "ERROR: %s\n", str);
	system("PAUSE");
	exit(1);
}

void swap(cl_mem* a, cl_mem* b) {
	cl_mem* tmp = a;
	a = b;
	b = tmp;
}
void swap(cl_mem a, cl_mem b) {
	cl_mem tmp = a;
	a = b;
	b = tmp;
}

int calc_nwg(int nels, int lws)
{
	int nwg = (nels / 2 + lws - 1) / lws;
	// if more than one workgroup, number must be even
	return nwg + (nwg > 1 && (nwg & 1));
}


int matrix_to_array_indexes(int i, int j, int row_len) {
	return i * row_len + j;
}

bool isEven(int a) {
	return (a % 2) == 0;
}

bool isEmpty(const cl_int4* v, int len, cl_int4 default_v) {
	for (int i = 0; i < len; i++) {
		if (v[i].s0 != default_v.s0 || v[i].s1 != default_v.s1 || v[i].s2 != default_v.s2 || v[i].s3 != default_v.s3)
			return false;
	}
	return true;
}

bool isEmpty(cl_int8* v, int len, cl_int8 default_v) {
	for (int i = 0; i < len; i++) {
		if (v[i].s0 != default_v.s0 || v[i].s1 != default_v.s1 || v[i].s2 != default_v.s2 || v[i].s3 != default_v.s3
			|| v[i].s4 != default_v.s4 || v[i].s5 != default_v.s5 || v[i].s6 != default_v.s6 || v[i].s7 != default_v.s7)
			return false;
	}
	return true;
}

bool isEmpty(int* v, int len, int default_v) {
	for (int i = 0; i < len; i++)
	{
		if (v[i] != default_v)
			return false;
	}
	return true;
}

bool is_file_empty(FILE* fp) {
	fseek(fp, 0, SEEK_END);
	int size = ftell(fp);
	rewind(fp);
	return size == 0;
}

size_t GetMetricsArrayLenght(int n_nodes) {
	return pow(2, ceil(log(n_nodes) / log(2)));
}



//void print(cl_int2* v, int len, const char* separator, bool withIndexes, int startingFrom) {
//	for (int i = startingFrom; i < len; i++)
//	{
//		if (withIndexes)
//			std::cout << i << ":";
//		std::cout << "(" << v[i].s[0] << ", " << v[i].s[1] << ")" << separator;
//	}
//	std::cout << "\n";
//}

void print(cl_int3* v, int len, const char* separator, bool withIndexes, int startingFrom) {
	for (int i = startingFrom; i < len; i++)
	{
		if (withIndexes)
			std::cout << i << ":";
		std::cout << "(" << v[i].s[0] << ", " << v[i].s[1] << ", " << v[i].s[2] << ")" << separator;
	}
	std::cout << "\n";
}

void print(cl_int4* v, int len, const char* separator, bool withIndexes) {
	for(int i = 0; i < len; i++)
	{
		if (withIndexes)
			std::cout << i << ":";
		std::cout << "(" << v[i].s[0] << ", " << v[i].s[1] << ", " << v[i].s[2] << ", " << v[i].s[3] << ")" << separator;
	}
	std::cout << "\n";
}

void print8(cl_int8* v, int len, const char* separator, bool withIndexes) {
	for(int i = 0; i < len; i++)
	{
		if (withIndexes)
			std::cout << i << ":";
		std::cout << "(" << v[i].s[0] << ", " << v[i].s[1] << ", " << v[i].s[2] << ", " << v[i].s[3] << ", " << v[i].s[4] << ", " << v[i].s[5] << ", " << v[i].s[6] << ", " << v[i].s[7] << ")" << separator;
	}
	std::cout << "\n";
}



/// <summary>
/// TODO: Print only if debug enabled and add to the .h file
/// </summary>
//int printd(char const* const _Format, ...) {
//	int _Result;
//	va_list _ArgList;
//	__crt_va_start(_ArgList, _Format);
//	_Result = _vfprintf_l(stdout, _Format, NULL, _ArgList);
//	__crt_va_end(_ArgList);
//	return _Result;
//}


//Una l è minore di r se hanno lo stesso livello, e l ha peso maggiore o se l ha livello minore di r.
bool operator<(const metrics_t& l, const metrics_t& r) {
	if (l.y == r.y) {
		return l.x > r.x; //peso maggiore
	}
	return l.y < r.y; //o livello più basso
}
bool operator>(const metrics_t& l, const metrics_t& r) {
	return (r < l);
}
bool operator<=(const metrics_t& l, const metrics_t& r) {
	return !(l > r);
}
bool operator>=(const metrics_t& l, const metrics_t& r) {
	return !(l < r);
}


#if WINDOWS
#define PIPEOPEN _popen
#else
#define PIPEOPEN popen
#endif // WINDOWS

#if WINDOWS
#define PIPECLOSE _pclose
#else
#define PIPECLOSE pclose
#endif // WINDOWS

std::string exec(const char* cmd) {
	char buffer[128];
	std::string result = "";
	FILE* pipe = PIPEOPEN(cmd, "r");
	if (!pipe) throw std::runtime_error("popen() failed!");
	try {
		while (fgets(buffer, sizeof buffer, pipe) != NULL) {
			result += buffer;
		}
	}
	catch (...) {
		PIPECLOSE(pipe);
		throw;
	}
	PIPECLOSE(pipe);
	return result;
}


void printMemoryUsage() {
#if WINDOWS
	//virtual memory used by program
	PROCESS_MEMORY_COUNTERS_EX pmc;
	GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
	SIZE_T virtualMemUsedByMe = pmc.PrivateUsage; //in bytes

	//ram used by program
	SIZE_T physMemUsedByMe = pmc.WorkingSetSize; //in bytes

	printf("Memory usage: %dMB, %dMB \n", virtualMemUsedByMe / 1000 / 1000, physMemUsedByMe / 1000 / 1000);
#endif
}