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
##********************************************************************/

include ../../../gpu_common.mk

TEST = test_choose
BENCHMARK = benchmark_choose

#-------------------------------------------------------------------------------
# Optional command line arguments
#-------------------------------------------------------------------------------



#benchmark block of threads size
ifdef blockSize
	BLOCK_SIZE = $(blockSize)
	DEFINES += -DRD_BLOCK_SIZE=$(BLOCK_SIZE)
	TEST_SUFFIX = $(BLOCK_SIZE)
endif

BENCHMARK_NAME = $(BENCHMARK)_$(TEST_SUFFIX)_$(BIN_SUFFIX)
TEST_NAME = $(TEST)_$(BIN_SUFFIX)

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

DEPS =				$(RD_DEPS)


all: build_test build_benchmark

build_test: $(TEST_NAME)
build_benchmark: $(BENCHMARK_NAME)

#----------------------------------------------------------------------

$(TEST_NAME): $(TEST).cu $(DEPS)
	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ $< $(LIBS)

#----------------------------------------------------------------------

# $(BENCHMARK_NAME): $(BENCHMARK).cu $(DEPS)
# 	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o bin/$@ $< $(LIBS)

#----------------------------------------------------------------------

BLOCK_SIZE_NUMBERS := $(shell seq 64 32 1024)
JOBS_PREFIX := $(addprefix $(BENCHMARK)_,$(BLOCK_SIZE_NUMBERS))
JOBS := $(addsuffix _$(BIN_SUFFIX),$(JOBS_PREFIX))
RUN_JOBS := $(addprefix run_,$(JOBS))

.PHONY: build_benchmark_thorough $(JOBS);
	echo $@ Success

build_benchmark_thorough: $(JOBS)

# $* is equal to what '%' match in current target
currBlockSize = $(firstword $(subst _, ,$*))

.PHONY: $(JOBS)
$(JOBS): $(BENCHMARK)%: $(BENCHMARK).cu $(DEPS)
	$(NVCC) $(DEFINES) -DRD_BLOCK_SIZE=$(currBlockSize) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) \
	$(INC) -o bin/$@ $< $(LIBS)

#----------------------------------------------------------------------

clean:
	rm -f *.o *_x86_64_* bin/*_x86_64_* *.png

run_test: 
	set_size=10; \
	for i in $(shell seq 1 6); do \
		./$(TEST_NAME) --np=$$set_size --r1=5.6 --r2=10.3 --a=22.4 --b=10.73 --s=4.42 || exit 1; \
		set_size=$$((set_size * 10)) ; \
	done

# run_benchmark_thorough: 
# 	for bname in $(JOBS); do \
# 		if [ "$$(bname)" = "$(firstword $(JOBS))" ]; then \
# 			./bin/$$(bname) --np=100000 --r1=5.6 --r2=11.2 --a=22.4 --b=10.73 --s=4.32 --log --start --d=$(device) || exit 1; \
# 		elif [ "$$(bname)" = "$(lastword $(JOBS))" ]; then \
# 			./bin/$$(bname) --np=100000 --r1=5.6 --r2=11.2 --a=22.4 --b=10.73 --s=4.32 --log --end --d=$(device) || exit 1; \
# 		else \
# 			./bin/$$(bname) --np=100000 --r1=5.6 --r2=11.2 --a=22.4 --b=10.73 --s=4.32 --log --d=$(device) || exit 1; \
# 		fi;	\
# 	done

run_benchmark_thorough: $(RUN_JOBS)

bname = $(patsubst run_%,%,$@)

.PHONY: $(RUN_JOBS)
$(RUN_JOBS):
	if [ "$@" = "$(firstword $(RUN_JOBS))" ]; then \
		./bin/$(bname) --np=100000 --r1=5.6 --r2=11.2 --a=22.4 --b=10.73 --s=4.32 --log --start --d=$(device) || exit 1; \
	elif [ "$@" = "$(lastword $(RUN_JOBS))" ]; then \
		./bin/$(bname) --np=100000 --r1=5.6 --r2=11.2 --a=22.4 --b=10.73 --s=4.32 --log --end --d=$(device) || exit 1; \
	else \
		./bin/$(bname) --np=100000 --r1=5.6 --r2=11.2 --a=22.4 --b=10.73 --s=4.32 --log --d=$(device) || exit 1; \
	fi;

