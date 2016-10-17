#
# @file Makefile
# @author Adam Rogowiec
#
# This file is an integral part of the master thesis entitled:
# "Elaboration and implementation in CUDA technology parallel version of 
#  estimation of multidimensional random variable density function ridge
#  detection algorithm.",
# which is supervised by prof. dr hab. inż. Marek Nałęcz.
# 
# Institute of Control and Computation Engineering
# Faculty of Electronics and Information Technology
# Warsaw University of Technology 2016
#

include ../../../cpu_common.mk

#-------------------------------------------------------------------------------
# Commandline Additional Options
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

#-------------------------------------------------------------------------------
# input/output file names
#-------------------------------------------------------------------------------

TEST := test_tiled_tree
TINY_BENCHMARK := benchmark_tiled_tree
HUGE_BENCHMARK := benchmark_tiled_tree2

SUFFIX = $(BIN_SUFFIX)_$(TEST_SUFFIX)_$(PRECISION_SUFFIX)_$(SCALING_SUFFIX)

TEST_NAME = $(TEST)_$(SUFFIX)
TINY_BENCHMARK_NAME = $(TINY_BENCHMARK)_$(SUFFIX)
HUGE_BENCHMARK_NAME = $(HUGE_BENCHMARK)_$(SUFFIX)

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

DEPS =      $(RD_DEPS) 

#-------------------------------------------------------------------------------
# Targets
#-------------------------------------------------------------------------------

all: build_test build_tiny_benchmark build_huge_benchmark

build_test: $(TEST_NAME)
build_tiny_benchmark: $(TINY_BENCHMARK_NAME)
build_huge_benchmark: $(HUGE_BENCHMARK_NAME)

#-------------------------------------------------------------------------------

$(TEST_NAME): $(TEST).cpp $(DEPS) 
	$(CC) $(DEFINES) $(CCFLAGS) $(INC) -o $@ $< $(LIBS)

#-------------------------------------------------------------------------------

$(TINY_BENCHMARK_NAME): $(TINY_BENCHMARK).cpp $(DEPS) 
	$(CC) $(DEFINES) $(CCFLAGS) $(INC) -o $@ $< $(LIBS)

#-------------------------------------------------------------------------------

$(HUGE_BENCHMARK_NAME): $(HUGE_BENCHMARK).cpp $(DEPS) 
	$(CC) $(DEFINES) $(CCFLAGS) $(INC) -o $@ $< $(LIBS)

#-------------------------------------------------------------------------------

run_dim_scaling: $(HUGE_BENCHMARK_NAME)
	./$(HUGE_BENCHMARK_NAME) --size=0 --segl=100 --a=0 --b=0 --stddev=2.17 --log

clean:
	rm -f *.o *x86_64* *.png

clean-images:
	rm -f *.png
    