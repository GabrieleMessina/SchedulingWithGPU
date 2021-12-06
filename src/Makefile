# CC=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30037\bin\Hostx86\x86\cl.exe
# nel caso dovesse smettere di funzionare la compilazione, potrebbe essere che CL.exe non viene più trovato, 
# potrebbe essersi aggiornato e quindi la cartella è cambiata... aggiorna il path o aggiunti il path qui stesso.
CC=cl.exe
CXX=cl.exe

INC_DIR = $(OPENCL_DIR)
INC_LIB = $(OPENCL_LIB_X86)\OpenCL.lib
OUT_DIR = $(@D)\build\\
# per info https://www.gnu.org/software/make/manual/make.html#Automatic-Variables
# w per disabilitare i warning
# Ot per ottimizzare il codice per le prestazioni e non per la grandezza dell'exe
# I per indicare la cartella da cui prendere gli include
# Fe per scegliere la cartella in cui gli exe saranno messi
# Fo per scegliere la cartella in cui gli obj, file intermedi, saranno messi
CFLAGS=/w /EHsc -Ot -I$(INC_DIR) #/Fo$(OUT_DIR) /Fe$(OUT_DIR)
CXXFLAGS=-w /EHsc -I$(INC_DIR) #/Fo$(OUT_DIR) /Fe$(OUT_DIR)
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
all: graph_init generator
# all: oclinfo QuicksortMain

.PHONY: all

