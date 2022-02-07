ODIR = obj
CXX = g++


OPENCL_DIR = /mnt/c/Users/gabry/Universita/GPGPU/IntelSWTools/sw_dev_tools/OpenCL/sdk/include
#OPENCL_LIB_X86 = /mnt/c/Users/gabry/Universita/GPGPU/IntelSWTools/sw_dev_tools/OpenCL/sdk/lib/x86
OPENCL_LIB_X86 = /usr/lib/x86_64-linux-gnu

INC_DIR = $(OPENCL_DIR)
INC_LIB = $(OPENCL_LIB_X86)/libOpenCL.so
OUT_DIR = $(@D)/build/

SOURCES = $(wildcard *.cpp)
SOURCES := $(filter-out generator.cpp, $(SOURCES))
OBJECTS = $(SOURCES:.cpp=.o)
HEADERS = $(wildcard *.hpp)

LDFLAGS = $(INC_LIB)
CXXFLAGS = -g -w -I$(INC_DIR) -L$(OPENCL_LIB_X86) -lOpenCL

default: ${OBJECTS} 
	${CXX} ${CXXFLAGS} ${OBJECTS} ${HEADERS] ${SOURCES} -o $(ODIR)/${PROGRAM}
clean: make rm ${OBJECTS} ${PROGRAM}