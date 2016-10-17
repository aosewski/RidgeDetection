#/***************************************************************************
# * Author: Adam Rogowiec
# *
# ***************************************************************************/

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

BENCHMARK = benchmark_dist_mtx

BENCHMARK_NAME = $(BENCHMARK)_$(BIN_SUFFIX)_$(TEST_SUFFIX)_$(TEST_SUFFIX2)

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

DEPS =				$(RD_DEPS)

# ------------------------------------------------------------------------------

build_benchmark: $(BENCHMARK_NAME)
build_cubin: $(BENCHMARK_NAME)_cubin
build_sass_w_code: $(BENCHMARK_NAME)_sass_with_code
build_sass_lrm: $(BENCHMARK_NAME)_sass_lrm

# ------------------------------------------------------------------------------
# %.o: %.cu $(DEPS)
# 	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ -c $<

#----------------------------------------------------------------------

$(BENCHMARK_NAME): $(BENCHMARK).cu $(DEPS) 
	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ $< $(LIBS)

#----------------------------------------------------------------------

$(BENCHMARK_NAME)_cubin: $(BENCHMARK).cu 
	$(NVCC) $(DEFINES) -arch=sm_35 $(NVCCFLAGS) $(CPU_ARCH) $(INC) \
	-cubin -o $(BENCHMARK_NAME).cubin $<

$(BENCHMARK_NAME)_sass_with_code: $(BENCHMARK_NAME).cubin
	nvdisasm -g -sf $(BENCHMARK_NAME).cubin > $(BENCHMARK_NAME)_sass_w_code.sass

$(BENCHMARK_NAME)_sass_lrm: $(BENCHMARK_NAME).cubin
	nvdisasm -lrm narrow -sf -g $(BENCHMARK_NAME).cubin > $(BENCHMARK_NAME)_sass_lrm.sass

clean:
	rm -f *.o *_x86_64_* *.png

