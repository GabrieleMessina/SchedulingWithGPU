CC=cl.exe
CXX=cl.exe

INC_DIR = $(OPENCL_DIR)
INC_LIB = $(OPENCL_LIB_X86)\OpenCL.lib
CFLAGS=/w -Ot -I$(INC_DIR)
CXXFLAGS=-w -I$(INC_DIR)
LDFLAGS = $(INC_LIB)

# all: vecinit_cl vecinit_wg_fix vecinit_preferred_wg vecinit_map_mem vecinit_more_work vecinit_vec vecinit_vec_coal
# all: vecsum_vec
# all: vecsmooth vecsmooth_lmem vecsmooth_vec vecsmooth_vec_lmem
# all: matinit transpose transpose_lmem transpose_lmem_pad transpose_lmem_x
# all: transpose_lmem_pitch transpose_lmem_pitch_rect
# all: transpose_img
# all: reduction reduction_lmem reduction_sat
# all: scan0 scan_lmem scan_lmem4
# all: imgtest
all: vecinit_cl
# all: oclinfo

.PHONY: all

