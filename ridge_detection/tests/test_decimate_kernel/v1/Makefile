#/***************************************************************************
# * Author: Adam Rogowiec
# *
# ***************************************************************************/

include ../../../gpu_common.mk


#-------------------------------------------------------------------------------
# Optional command line arguments
#-------------------------------------------------------------------------------

#benchmark block of threads size
ifdef blockSize
	BLOCK_SIZE = $(blockSize)
	DEFINES += -DRD_BLOCK_SIZE=$(BLOCK_SIZE)
	TEST_SUFFIX = $(BLOCK_SIZE)
endif

#-------------------------------------------------------------------------------
# input/output file names
#-------------------------------------------------------------------------------

TEST = test_decimate
BENCHMARK = benchmark_decimate

BENCHMARK_NAME = $(BENCHMARK)_$(TEST_SUFFIX)_$(BIN_SUFFIX)
TEST_NAME = $(TEST)_$(BIN_SUFFIX)


#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

DEPS =		$(RD_DEPS)


all: build_test build_benchmark

build_test: $(TEST_NAME) 
build_benchmark: $(BENCHMARK_NAME)

# ------------------------------------------------------------------------------

# %.o: %.cu $(DEPS)
# 	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ -c $<

# ------------------------------------------------------------------------------

$(TEST_NAME): $(TEST).cu $(DEPS)
	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o bin/$@ $< $(LIBS)

# ------------------------------------------------------------------------------

# $(BENCHMARK_NAME): $(BENCHMARK).cu $(DEPS)
# 	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o bin/$@ $< $(LIBS)

# ------------------------------------------------------------------------------

BLOCK_SIZE_NUMBERS := $(shell seq 64 32 1024)
JOBS_PREFIX := $(addprefix $(BENCHMARK)_,${BLOCK_SIZE_NUMBERS})
JOBS := $(addsuffix _$(BIN_SUFFIX),${JOBS_PREFIX})

.PHONY: benchmark_thorough ${JOBS}
	echo $@ Success
	
benchmark_thorough: ${JOBS}

# $* is equal to what '%' match in current target
currBlockSize = $(firstword $(subst _, ,$*))

.PHONY: ${JOBS}
${JOBS}: $(BENCHMARK)%: $(BENCHMARK).cu $(DEPS)
	$(NVCC) $(DEFINES) -DRD_BLOCK_SIZE=$(currBlockSize) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o bin/$@ $< $(LIBS)

#----------------------------------------------------------------------

clean:
	rm -f *.o *_x86_64_* bin/*_x86_64_* *.png

run_test: 
	set_size=10; \
	for i in $(shell seq 1 6); do \
		./$(TEST_NAME) --np=$$set_size --r1=5.6 --r2=10.3 --a=22.4 --b=10.73 --s=4.42 || exit 1; \
		set_size=$$((set_size * 10)) ; \
	done

run_benchmark_thorough: 
	for bname in $(JOBS); do \
		if [ "$${bname}" = "$(firstword $(JOBS))" ]; then \
			./bin/$${bname} --np=250000 --r1=5.6 --r2=11.2 --a=22.4 --b=10.73 --s=4.32 --log --start --d=$(device) || exit 1; \
		elif [ "$${bname}" = "$(lastword $(JOBS))" ]; then \
			./bin/$${bname} --np=250000 --r1=5.6 --r2=11.2 --a=22.4 --b=10.73 --s=4.32 --log --end --d=$(device) || exit 1; \
		else \
			./bin/$${bname} --np=250000 --r1=5.6 --r2=11.2 --a=22.4 --b=10.73 --s=4.32 --log --d=$(device) || exit 1; \
		fi;	\
	done


