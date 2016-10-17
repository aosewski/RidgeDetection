#***************************************************************************
# Author: Adam Rogowiec
#
#***************************************************************************/

include ../../gpu_common.mk

TEST = test_dynamic_vector
SIMPLE = test_simple
#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

DEPS =              $(RD_DEPS)


all: build_test build_simple

build_test: $(TEST)
build_simple: $(SIMPLE)

%.o: %.cu $(DEPS)
	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ -c $<

#----------------------------------------------------------------------

$(TEST): $(TEST).o 
	$(NVCC) $(SM_TARGETS) $(CPU_ARCH) $(NVCCFLAGS) -o $@ $+ $(LIBS)

#----------------------------------------------------------------------

$(SIMPLE): $(SIMPLE).o 
	$(NVCC) $(SM_TARGETS) $(CPU_ARCH) $(NVCCFLAGS) -o $@ $+ $(LIBS)

#----------------------------------------------------------------------

clean:
	rm -f *.o $(TEST) $(SIMPLE) *.png


