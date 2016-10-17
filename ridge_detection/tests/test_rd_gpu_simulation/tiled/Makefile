##
## @file Makefile
## @author     Adam Rogowiec
##
## This file is an integral part of the master thesis entitled:
## "Elaboration and implementation in CUDA technology parallel version of
## estimation of multidimensional random variable density function ridge
## detection algorithm."
## , which is conducted under the supervision of prof. dr hab. inż. Marek
## Nałęcz.
##
## Institute of Control and Computation Engineering Faculty of Electronics and
## Information Technology Warsaw University of Technology 2016
##

include ../../../gpu_common.mk

#-------------------------------------------------------------------------------
# Optional command line arguments
#-------------------------------------------------------------------------------

# Testing mode option (quick/thorough)
ifeq ($(quickertest), 1)
	DEFINES += -DQUICKER_TEST
	TEST_SUFFIX = quicker
else ifeq ($(quicktest), 1)
	DEFINES += -DQUICK_TEST
	TEST_SUFFIX = quick
else 
	TEST_SUFFIX = thorough
endif

#drawing tree tiles
ifeq ($(drawTiles), 1)
	DEFINES += -DRD_DRAW_TREE_TILES
endif

#inner ridge detection kernel timing
ifeq ($(inkernelTiming), 1)
	DEFINES += -DRD_INNER_KERNEL_TIMING
endif

# benchmark only double precision 
ifdef dprec
	DEFINES += -DRD_DOUBLE_PRECISION
	PRECISION_SUFFIX = dprec
else
	PRECISION_SUFFIX = sprec
endif

ifdef testScaling
	DEFINES += -DRD_TEST_SCALING
	SCALING_SUFFIX = scaling
else
	SCALING_SUFFIX = noscaling
endif


SUFFIX = $(TEST_SUFFIX)_$(PRECISION_SUFFIX)_$(SCALING_SUFFIX)

#-------------------------------------------------------------------------------
# input/output file names
#-------------------------------------------------------------------------------

BENCHMARK = benchmark
BENCHMARK_NAME = $(BENCHMARK)_$(BIN_SUFFIX)_$(SUFFIX)

OUTPUT_BIN_DIR = bin

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

DEPS =  $(RD_DEPS)\
        $(BASE_DIR)tests/test_util.hpp


#-------------------------------------------------------------------------------
# Targets
#-------------------------------------------------------------------------------

all: build_benchmark
build_benchmark: $(BENCHMARK_NAME)

# $(TARGET).o: $(TARGET).cu
#   $(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ -c $^

# $(TARGET): $(TARGET).o
#   $(NVCC) $(SM_TARGETS) $(NVCCFLAGS) -o $@ $+ $(LIBS)

#-------------------------------------------------------------------------------

$(BENCHMARK_NAME): $(BENCHMARK).cu $(DEPS)
	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $(OUTPUT_BIN_DIR)/$@ $< $(LIBS)

#-------------------------------------------------------------------------------

TEST_DIM_NUMBERS := $(shell seq 2 1 12)
JOBS_PREFIX1 := $(addprefix $(BENCHMARK)_,$(TEST_DIM_NUMBERS))
JOBS_PREFIX2 := $(addsuffix _DIM_$(BIN_SUFFIX)_,$(JOBS_PREFIX1))
JOBS := $(addsuffix $(SUFFIX),$(JOBS_PREFIX2))

.PHONY: benchmark_dim ${JOBS}
	echo $@ Success
	
benchmark_dim: ${JOBS}

# $* is equal to what '%' match in current target
currTestDim = $(firstword $(subst _, ,$*))

.PHONY: ${JOBS}
${JOBS}: $(BENCHMARK)%: $(BENCHMARK).cu $(DEPS)
	$(NVCC) $(DEFINES) -DRD_TEST_DIMENSION=$(currTestDim) $(SM_TARGETS) $(NVCCFLAGS) \
	$(CPU_ARCH) $(INC) $< $(LIBS) -o $(OUTPUT_BIN_DIR)/$@

#----------------------------------------------------------------------

# run_benchmark_dim: 
# 	for bname in $(JOBS); do \
# 		./$(OUTPUT_BIN_DIR)/$${bname} --size=10000000 --segl=100 --a=0 --b=0 --stddev=2.17 --log \
# 		--f=../../data/segment_pc_np=1e7_segl=100_stddev=2.17_d=12.bin --fd=12 \
# 		--d=$(device) || exit 1; \
# 	done

run_benchmark_dim: 
	for bname in $(JOBS); do \
		./$(OUTPUT_BIN_DIR)/$${bname} --size=10000000 --segl=0 --a=22.52 --b=11.31 --stddev=4.17 --log \
		--f=../../data/spiral_pc_np=1e7_a=22.52_b=11.31_stddev=4.17_d=3.bin --fd=3 \
		--d=$(device) || exit 1; \
	done

# --size=10000000 --a=22.52 --b=11.31 --stddev=4.17 --log --f=../../data/spiral_pc_np=1e7_a=22.52_b=11.31_stddev=4.17_d=3.bin --fd=3 --dim=2
#-------------------------------------------------------------------------------

clean:
	rm -f *.o  *_x86_64_* *.png 
	rm -f *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx *.hash *.cu.cpp *.o *.fatbin *.module_id *.reg.c

clean-images:
	rm -f *.png 
    
print: 
	\echo 'Zaleznosci: $(DEPS)'
    
    
# #(CC) -MM  kompilacja z flagą MM wypluwa wszystkie zależności 
# include -dependecies