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

include ../../cpu_common.mk

#-------------------------------------------------------------------------------
# Optional command line arguments
#-------------------------------------------------------------------------------

# benchmark only double precision 
ifdef dprec
	DEFINES += -DRD_DOUBLE_PRECISION
	PRECISION_SUFFIX = dprec
else
	PRECISION_SUFFIX = sprec
endif


SUFFIX = $(PRECISION_SUFFIX)

#-------------------------------------------------------------------------------
# input/output file names
#-------------------------------------------------------------------------------

TARGET := rd_mr
BENCHMARK := benchmark
BENCHMARK_NAME = $(BENCHMARK)_$(BIN_SUFFIX)_$(SUFFIX)

OUTPUT_BIN_DIR = bin

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

DEPS :=     $(RD_DEPS)

MR_DIR := $(BASE_DIR)rd/cpu/mrupniewski

MR_DEPS :=     	$(MR_DIR)/choose.hpp 	\
				$(MR_DIR)/decimate.hpp 	\
				$(MR_DIR)/evolve.hpp 	\
				$(MR_DIR)/order.hpp 	\
				$(MR_DIR)/sglib.h 	

LIBS += -lgsl -lgslcblas -lm

#-------------------------------------------------------------------------------
# Targets
#-------------------------------------------------------------------------------


all: build

build: $(TARGET)
build_benchmark: $(BENCHMARK_NAME)

$(TARGET): $(TARGET).cpp $(MR_DEPS) $(DEPS)
	$(CC) $(DEFINES) $(CCFLAGS) $(INC) -o $@ $< $(LIBS)

#-------------------------------------------------------------------------------

$(BENCHMARK_NAME): $(BENCHMARK).cpp $(MR_DEPS) $(DEPS)
	$(CC) $(DEFINES) $(CCFLAGS) $(INC) -o $(OUTPUT_BIN_DIR)/$@ $< $(LIBS)

#-------------------------------------------------------------------------------

TEST_DIM_NUMBERS := $(shell seq 2 1 6)
JOBS_PREFIX1 := $(addprefix $(BENCHMARK)_,$(TEST_DIM_NUMBERS))
JOBS_PREFIX2 := $(addsuffix _DIM_$(BIN_SUFFIX)_,$(JOBS_PREFIX1))
JOBS := $(addsuffix $(SUFFIX),$(JOBS_PREFIX2))
RUN_JOBS := $(addprefix run_,$(JOBS))

.PHONY: benchmark_dim ${JOBS}
	echo $@ Success
	
benchmark_dim: ${JOBS}

# $* is equal to what '%' match in current target
currTestDim = $(firstword $(subst _, ,$*))

.PHONY: ${JOBS}
${JOBS}: $(BENCHMARK)%: $(BENCHMARK).cpp $(MR_DEPS) $(DEPS)
	$(CC) $(DEFINES) -DRD_TEST_DIM=$(currTestDim) $(CCFLAGS) \
	$(INC) $< $(LIBS) -o $(OUTPUT_BIN_DIR)/$@

#----------------------------------------------------------------------

# run_benchmark_dim: 
# 	for bname in $(JOBS); do \
# 		./$(OUTPUT_BIN_DIR)/$${bname} --size=10000000 --segl=100 --a=0 --b=0 --stddev=2.17 --log \
# 		--f=../data/segment_pc_np=1e7_segl=100_stddev=2.17_d=12.bin --fd=12 --dim=0\
# 		|| exit 1; \
# 	done

#-------------------------------------------------------------------------------

run_benchmark_dim: $(RUN_JOBS)
bname = $(patsubst run_%,%,$@)
currTestDim2 = $(word 2,$(subst _, ,$(bname)))

# --f=../data/segment_pc_np=1e7_segl=100_stddev=2.17_d=12.bin --fd=12
# --f=../data/spiral_pc_np=1e7_a=22.52_b=11.31_stddev=4.17_d=3.bin --fd=3

.PHONY: $(RUN_JOBS)
# $(RUN_JOBS): $(JOBS)
# 	./$(OUTPUT_BIN_DIR)/$(bname) --size=10000000 --segl=100 --stddev=2.17 --log \
# 	--f=../data/segment_pc_np=1e7_segl=100_stddev=2.17_d=12.bin --fd=12 --dim=$(currTestDim2) \
# 	|| exit 1;
$(RUN_JOBS): $(JOBS)
	./$(OUTPUT_BIN_DIR)/$(bname) --size=10000000 --a=22.52 --b=11.31 --stddev=4.17 --log \
	--f=../data/spiral_pc_np=1e7_a=22.52_b=11.31_stddev=4.17_d=3.bin --fd=3 --dim=$(currTestDim2) \
	|| exit 1;


# --size=10000000 --a=22.52 --b=11.31 --stddev=4.17 --log --f=../data/spiral_pc_np=1e7_a=22.52_b=11.31_stddev=4.17_d=3.bin --fd=3 --dim=2 --rGraphs
# --size=10000000 --segl=100 --stddev=2.17 --log --f=../data/segment_pc_np=1e7_segl=100_stddev=2.17_d=12.bin --fd=12

#-------------------------------------------------------------------------------

clean:
	rm -rf *.o $(TARGET) *.png

print: 
	@echo 'rwildcard: $(rwildcard)'
	@echo 'BASE_DIR: $(BASE_DIR)'
	@echo 'Zaleznosci: $(DEPS)'
	@echo 'Zaleznosci: $(MR_DEPS)'
