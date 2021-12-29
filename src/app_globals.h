#pragma once

#define DEBUG_DAG_INIT false
#define DEBUG_ENTRY_DISCOVER false
#define DEBUG_COMPUTE_METRICS false
#define DEBUG_SORT false


#ifdef _DEBUG
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
// Replace _NORMAL_BLOCK with _CLIENT_BLOCK if you want the
// allocations to be of _CLIENT_BLOCK type
#else
#define DBG_NEW new
#endif
