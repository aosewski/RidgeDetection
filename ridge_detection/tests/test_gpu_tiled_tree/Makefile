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


TEST = test_tiled_tree
DUMMY2 = dummy_test2
FOO = foo

TEST_NAME = $(TEST)_$(BIN_SUFFIX)_$(TEST_SUFFIX)_$(TEST_SUFFIX2)

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

DEPS =  $(RD_DEPS)\
        $(BASE_DIR)tests/test_util.hpp

#-------------------------------------------------------------------------------
# Targets
#-------------------------------------------------------------------------------

all: build_test 
build_test: $(TEST_NAME)

# %.o: %.cu
#   $(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ -c $^

# $(TARGET): $(TARGET).o
#   $(NVCC) $(SM_TARGETS) $(NVCCFLAGS) -o $@ $+ $(LIBS)

#-------------------------------------------------------------------------------

$(TEST_NAME): $(TEST).cu $(DEPS)
	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ $< $(LIBS)

#-------------------------------------------------------------------------------

$(DUMMY2): $(DUMMY2).cu $(DEPS)
	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ $< $(LIBS)

$(FOO): $(FOO).cu $(DEPS)
	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ $< $(LIBS)

#-------------------------------------------------------------------------------

# $(BENCHMARK): $(BENCHMARK).cu $(DEPS)
#     $(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ $< $(LIBS)

#-------------------------------------------------------------------------------

clean:
	rm -f *.o  *_x86_64_* $(DUMMY2) $(FOO) *.png 
	rm -f *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx *.hash *.cu.cpp *.o *.fatbin \
		*.module_id *.reg.c

clean-images:
	rm -f *.png 

print: 
	\echo 'Zaleznosci: $(DEPS)'
    
    
# #(CC) -MM  kompilacja z flagą MM wypluwa wszystkie zależności 
# include -dependecies