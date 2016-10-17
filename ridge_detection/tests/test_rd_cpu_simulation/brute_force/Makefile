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

include ../../../cpu_common.mk

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

# benchmark only double precision 
ifdef dprec
	DEFINES += -DRD_DOUBLE_PRECISION
	PRECISION_SUFFIX = dprec
else
	PRECISION_SUFFIX = sprec
endif

TEST_SUFFIX2 = $(PRECISION_SUFFIX)

#-------------------------------------------------------------------------------
# input/output file names
#-------------------------------------------------------------------------------

TEST := test_rd_cpu_simulation
BENCHMARK := benchmark

SUFFIX = $(BIN_SUFFIX)_$(TEST_SUFFIX)_$(TEST_SUFFIX2)
TEST_NAME = $(TEST)_$(SUFFIX)
BENCHMARK_NAME = $(BENCHMARK)_$(SUFFIX)

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

DEPS =		$(RD_DEPS)


#-------------------------------------------------------------------------------
# Targets
#-------------------------------------------------------------------------------

all: build

build_test: $(TEST_NAME)
build_benchmark: $(BENCHMARK_NAME)

#-------------------------------------------------------------------------------

$(TEST_NAME): $(TEST).cpp $(DEPS)
	$(CC) $(DEFINES) $(CCFLAGS) $(INC) -o $@ $< $(LIBS)

#-------------------------------------------------------------------------------

$(BENCHMARK_NAME): $(BENCHMARK).cpp $(DEPS)
	$(CC) $(DEFINES) $(CCFLAGS) $(INC) -o $@ $< $(LIBS)


#-------------------------------------------------------------------------------

clean:
	rm -f *.o *omp* *.png

clean-images:
	rm -f *.png 
	

