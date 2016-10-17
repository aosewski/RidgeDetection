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

# benchmark ccsc kernel
ifdef ccsc
	DEFINES += -DRD_CCSC
	KERNEL_SUFFIX = ccsc
endif

# benchmark stmc kernel
ifdef stmc
	DEFINES += -DRD_STMC
	KERNEL_SUFFIX = stmc
endif

# benchmark only double precision 
ifdef dprec
	DEFINES += -DRD_DOUBLE_PRECISION
	PRECISION_SUFFIX = dprec
else
	PRECISION_SUFFIX = sprec
endif

TEST_SUFFIX2 = $(KERNEL_SUFFIX)_$(PRECISION_SUFFIX)

#-------------------------------------------------------------------------------
# input/output file names
#-------------------------------------------------------------------------------

TEST_DEV = test_dev_evolve
BENCHMARK_DEV = benchmark_dev_evolve
BENCHMARK_LC = benchmark_lc_evolve


BENCHMARK_DEV_NAME = $(BENCHMARK_DEV)_$(BIN_SUFFIX)_$(TEST_SUFFIX)
TEST_DEV_NAME = $(TEST_DEV)_$(BIN_SUFFIX)_$(TEST_SUFFIX)
BENCHMARK_LC_NAME = $(BENCHMARK_LC)_$(BLOCK_SIZE)_$(BIN_SUFFIX)_$(TEST_SUFFIX)_$(TEST_SUFFIX2)

OUTPUT_BIN_DIR = bin2

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

DEPS =				$(RD_DEPS) \
					$(BASE_DIR)tests/test_evolve_brute_force/evolve_gold.hpp

# $(BASE_DIR)tests/test_evolve_brute_force/ut_evolve_base2.cuh \
# $(BASE_DIR)tests/test_evolve_brute_force/ut_evolve4.cuh \
# $(BASE_DIR)tests/test_evolve_brute_force/ut_profile_evolve4.cuh \


#-------------------------------------------------------------------------------
# Targets
#-------------------------------------------------------------------------------

all: build_benchmark build_test

build_benchmark: $(BENCHMARK_DEV_NAME)
build_test: $(TEST_DEV_NAME)

# ------------------------------------------------------------------------------
# %.o: %.cu $(DEPS)
# 	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ -c $<

# ------------------------------------------------------------------------------

$(TEST_DEV_NAME): $(TEST_DEV).cu $(DEPS)
	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ $< $(LIBS)

# ------------------------------------------------------------------------------

$(BENCHMARK_DEV_NAME): $(BENCHMARK_DEV).cu $(DEPS) 
	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ $< $(LIBS)

# ------------------------------------------------------------------------------

clean:
	rm -f *_x86_64_* *.png 
	rm -f *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx *.hash *.cu.cpp *.o *.fatbin \
	*.module_id *.reg.c

clean-img:
	rm -f *.png

clean-blc:
	rm -f $(OUTPUT_BIN_DIR)/*_x86_64_*


#----------------------------------------------------------------------

BLOCK_SIZE_NUMBERS := $(shell seq 64 32 1024)
JOBS_PREFIX1 := $(addprefix $(BENCHMARK_LC)_,$(BLOCK_SIZE_NUMBERS))
JOBS_PREFIX2 := $(addsuffix _$(BIN_SUFFIX)_,$(JOBS_PREFIX1))
JOBS := $(addsuffix $(TEST_SUFFIX)_$(TEST_SUFFIX2),$(JOBS_PREFIX2))

.PHONY: benchmark_lc_conf ${JOBS}
	echo $@ Success
	
benchmark_lc_conf: ${JOBS}

# $* is equal to what '%' match in current target
currBlockSize = $(firstword $(subst _, ,$*))

.PHONY: ${JOBS}
${JOBS}: $(BENCHMARK_LC)%: $(BENCHMARK_LC).cu $(DEPS)
	$(NVCC) $(DEFINES) -DRD_BLOCK_SIZE=$(currBlockSize) $(SM_TARGETS) $(NVCCFLAGS) \
	$(CPU_ARCH) $(INC) -o $(OUTPUT_BIN_DIR)/$@ $< $(LIBS)

#----------------------------------------------------------------------

run_test: test_dev
	set_size=10; \
	for i in $(shell seq 1 6); do \
		./$(TEST_DEV) --np=$$set_size --r1=5.6 --r2=10.3 --a=22.4 --b=10.73 --s=4.42 || exit 1; \
		set_size=$$((set_size * 10)) ; \
	done

## --np=500000 --r1=6.16 --r2=12.32 --a=1457.75 --b=0 --s=4.32
## --np=250000 --r1=5.6 --r2=11.2 --a=22.4 --b=10.73 --s=4.32
## --np=250000 --r1=8.15 --r2=16.3 --a=1457.74 --b=0 --s=4.32

run_benchmark_lc_conf: 
	for bname in $(JOBS); do \
		if [ "$${bname}" = "$(firstword $(JOBS))" ]; then \
			./$(OUTPUT_BIN_DIR)/$${bname} --np=250000 --r1=8.15 --r2=16.3 --a=1457.74 --b=0 --s=4.32 --log --start --d=$(device) || exit 1; \
		elif [ "$${bname}" = "$(lastword $(JOBS))" ]; then \
			./$(OUTPUT_BIN_DIR)/$${bname} --np=250000 --r1=8.15 --r2=16.3 --a=1457.74 --b=0 --s=4.32 --log --end --d=$(device) || exit 1; \
		else \
			./$(OUTPUT_BIN_DIR)/$${bname} --np=250000 --r1=8.15 --r2=16.3 --a=1457.74 --b=0 --s=4.32 --log --d=$(device) || exit 1; \
		fi;	\
	done

