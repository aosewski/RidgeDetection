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

include ../../gpu_common.mk

#-------------------------------------------------------------------------------
# Optional command line arguments
#-------------------------------------------------------------------------------

# Testing mode option (quick/thorough)
ifeq ($(quicktest), 1)
	DEFINES += -DQUICK_TEST
	TEST_SUFFIX += quick
else 
	TEST_SUFFIX += thorough
endif

#benchmark block of threads size
ifdef blockSize
	BLOCK_SIZE = $(blockSize)
	DEFINES += -DRD_BLOCK_SIZE=$(BLOCK_SIZE)
endif

#-------------------------------------------------------------------------------
# input/output file names
#-------------------------------------------------------------------------------

TEST = test_spatial_histogram
BENCHMARK_V4 = benchmark_spatial_histogram_v4

TEST_NAME = $(TEST)_$(BIN_SUFFIX)
BENCHMARK_V4_NAME = $(BENCHMARK_V4)_$(BLOCK_SIZE)_$(BIN_SUFFIX)_$(TEST_SUFFIX)

BENCHMARKS = 	$(BENCHMARK_V4_NAME)

OUTPUT_BIN_DIR = bin

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

DEPS =              $(RD_DEPS) \
					$(BASE_DIR)tests/test_util.hpp

#-------------------------------------------------------------------------------
# Targets
#-------------------------------------------------------------------------------

all: build_test build_benchmarks

build_test: $(TEST_NAME)
build_benchmarks: $(BENCHMARKS)

build_benchmark_v4: $(BENCHMARK_V4_NAME)

# %.o: %.cu $(DEPS)
# 	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ -c $<

#----------------------------------------------------------------------

$(TEST_NAME): $(TEST).cu $(DEPS) 
	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ $< $(LIBS)

#----------------------------------------------------------------------

$(BENCHMARK_V4_NAME): $(BENCHMARK_V4).cu $(DEPS) 
	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ $< $(LIBS)

#----------------------------------------------------------------------

clean:
	rm -f *.o *_x86_64_* *.png


#----------------------------------------------------------------------

BLOCK_SIZE_NUMBERS := 64 96 128 192 224 256 384 512
JOBS_PREFIX1 := $(addprefix $(BENCHMARK_V4)_,$(BLOCK_SIZE_NUMBERS))
JOBS_PREFIX2 := $(addsuffix _$(BIN_SUFFIX)_,$(JOBS_PREFIX1))
JOBS := $(addsuffix $(TEST_SUFFIX),$(JOBS_PREFIX2))

.PHONY: benchmark_lc_conf ${JOBS}
	echo $@ Success
	
benchmark_lc_conf: ${JOBS}

# $* is equal to what '%' match in current target
currBlockSize = $(firstword $(subst _, ,$*))

.PHONY: ${JOBS}
${JOBS}: $(BENCHMARK_V4)%: $(BENCHMARK_V4).cu $(DEPS)
	$(NVCC) $(DEFINES) -DRD_BLOCK_SIZE=$(currBlockSize) $(SM_TARGETS) $(NVCCFLAGS) \
	$(CPU_ARCH) $(INC) -o $(OUTPUT_BIN_DIR)/$@ $< $(LIBS)

#----------------------------------------------------------------------

run_benchmark_lc_conf: 
	for bname in $(JOBS); do \
		./$(OUTPUT_BIN_DIR)/$${bname} --size=1000000 --segl=100 --stddev=2.17 --log --d=$(device) || exit 1; \
	done
