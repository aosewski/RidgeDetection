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
ifeq ($(quickertest), 1)
	DEFINES += -DQUICKER_TEST
	TEST_SUFFIX += quicker
else ifeq ($(quicktest), 1)
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

ifeq ($(dblPrec), 1)
	DEFINES += -DRD_TEST_DBL_PREC
	TEST_PREFFIX := d
else
	TEST_PREFFIX := f
endif

#-------------------------------------------------------------------------------
# input/output file names
#-------------------------------------------------------------------------------

BENCHMARK_BF = benchmark_select_if_mem_bf
BENCHMARK_DV = benchmark_select_if_mem_dv
BENCHMARK_LC = benchmark_lc_select_if_mbf

BENCHMARK_BF_NAME = $(BENCHMARK_BF)_$(BIN_SUFFIX)_$(TEST_SUFFIX)
BENCHMARK_DV_NAME = $(BENCHMARK_DV)_$(BIN_SUFFIX)_$(TEST_SUFFIX)
BENCHMARK_LC_NAME = $(TEST_PREFFIX)_$(BENCHMARK_LC)_$(BLOCK_SIZE)_$(BIN_SUFFIX)_$(TEST_SUFFIX)

BIN_DIR = bin

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

DEPS =              $(RD_DEPS)	\
					$(BASE_DIR)tests/test_util.hpp


all: build_benchmark


build_benchmark_bf: $(BENCHMARK_BF_NAME)
build_benchmark_dv: $(BENCHMARK_DV_NAME)

#----------------------------------------------------------------------

# %.o: %.cu $(DEPS)
# 	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ -c $<
# $(TARGET): $(TARGET).o
# 	$(NVCC) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ $+ $(LIBS)

#----------------------------------------------------------------------

$(BENCHMARK_BF_NAME): $(BENCHMARK_BF).cu $(DEPS) 
	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ $< $(LIBS)

#----------------------------------------------------------------------

$(BENCHMARK_DV_NAME): $(BENCHMARK_DV).cu $(DEPS)
	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ $< $(LIBS)

#----------------------------------------------------------------------

clean:
	rm -f *_x86_64_* bin/*_x86_64_* *.png

clean-img:
	rm -f *.png

#----------------------------------------------------------------------

# BLOCK_SIZE_NUMBERS := $(shell seq 64 32 1024)
BLOCK_SIZE_NUMBERS := 64 96 128 192 224 256 384 512
JOBS_PREFIX1 := $(addprefix $(TEST_PREFFIX)_$(BENCHMARK_LC)_,${BLOCK_SIZE_NUMBERS})
JOBS_PREFIX2 := $(addsuffix _$(BIN_SUFFIX),${JOBS_PREFIX1})
JOBS := $(addsuffix _$(TEST_SUFFIX),${JOBS_PREFIX2})

.PHONY: benchmark_lc_thorough ${JOBS}
	echo $@ Success
	
benchmark_lc_thorough: ${JOBS}

# $* is equal to what '%' match in current target
currBlockSize = $(firstword $(subst _, ,$*))

.PHONY: ${JOBS}
${JOBS}: $(TEST_PREFFIX)_$(BENCHMARK_LC)%: $(BENCHMARK_LC).cu $(DEPS)
	$(NVCC) $(DEFINES) -DRD_BLOCK_SIZE=$(currBlockSize) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $(BIN_DIR)/$@ $< $(LIBS)

#----------------------------------------------------------------------

run_benchmark_thorough: 
	for bname in $(JOBS); do \
		if [ "$${bname}" = "$(firstword $(JOBS))" ]; then \
			./$(BIN_DIR)/$${bname} --np=250000 --ratio=0.6 --log --start --d=$(device) || exit 1; \
			./$(BIN_DIR)/$${bname} --np=250000 --ratio=0.2 --log --d=$(device) || exit 1; \
		elif [ "$${bname}" = "$(lastword $(JOBS))" ]; then \
			./$(BIN_DIR)/$${bname} --np=250000 --ratio=0.6 --log --end --d=$(device) || exit 1; \
			./$(BIN_DIR)/$${bname} --np=250000 --ratio=0.2 --log --d=$(device) || exit 1; \
		else \
			./$(BIN_DIR)/$${bname} --np=250000 --ratio=0.6 --log --d=$(device) || exit 1; \
			./$(BIN_DIR)/$${bname} --np=250000 --ratio=0.2 --log --d=$(device) || exit 1; \
		fi;	\
	done


