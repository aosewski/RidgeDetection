#/***************************************************************************
# * Author: Adam Rogowiec
# *
# ***************************************************************************/

include ../../gpu_common.mk

TEST = test_count_neighbours
BENCHMARK = benchmark_count_neighbours
TEST2 = test_count_dist_points

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

DEPS =		$(RD_DEPS)


all: build_test build_benchmark

build_test: $(TEST) 
build_test2: $(TEST2) 
build_benchmark: $(BENCHMARK)

# ------------------------------------------------------------------------------

# %.o: %.cu $(DEPS)
# 	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ -c $<

# ------------------------------------------------------------------------------

$(TEST): $(TEST).cu $(DEPS)
	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ $< $(LIBS)

# ------------------------------------------------------------------------------

$(TEST2): $(TEST2).cu $(DEPS)
	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ $< $(LIBS)

# ------------------------------------------------------------------------------

$(BENCHMARK): $(TEST).cu $(DEPS)
	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ $< $(LIBS)

# ------------------------------------------------------------------------------

test: $(TEST)
	./$(TEST) --np=1000 --r1=30 --s=2 --v

benchmark: $(BENCHMARK)
	./$(BENCHMARK) --np=100000 --r1=50 --r2=50 --a=45 --b=27 --s=20

profile: build_benchmark
	nvprof --profile-from-start off --print-gpu-trace ./$(BENCHMARK) --np=100000 --r1=50 --r2=50 --a=45 --b=27 --s=20

clean:
	rm -f *.o $(TEST) $(TEST2) $(BENCHMARK) *.png

clean-images:
	rm *.png

clobber:
	clean

print: 
	\echo 'Zaleznosci: $(DEPS)'

