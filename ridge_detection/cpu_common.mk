# @file cpu_common.mk
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

#-------------------------------------------------------------------------------
# Commandline Options
#-------------------------------------------------------------------------------

# [debug=<0|1>] Generate debug mode code
ifeq ($(debug), 1)
	CCFLAGS += -g3 -ggdb3 -Og
	DEFINES += -DRD_DEBUG
	BUILD_SUFFIX = dbg
else
	CCFLAGS += -O3
	LDFLAGS += -O3
	BUILD_SUFFIX = release
endif

# [omp=<0|1>] use openmp
ifeq ($(omp), 1)
	CCFLAGS += -fopenmp
	LDFLAGS += -fopenmp
	DEFINES += -DRD_USE_OPENMP
	OPENMP_SUFFIX = omp
else
	OPENMP_SUFFIX = noomp
endif

# [profile=<0|1>] profile 
ifeq ($(profile), 1)
	DEFINES += -DRD_PROFILE
	BUILD_SUFFIX = prof
endif

CC	 		:= /usr/bin/g++-5
CCFLAGS 	+= -std=c++11 -Wall -Wextra
LDFLAGS		+= -std=c++11 -Wall -Wextra

CPU_ARCH = x86_64
BIN_SUFFIX = $(CPU_ARCH)_$(OPENMP_SUFFIX)_$(BUILD_SUFFIX)

#-------------------------------------------------------------------------------
# Include/library directories and libraries variables
#-------------------------------------------------------------------------------

BASE_DIR = $(dir $(lastword $(MAKEFILE_LIST)))
RD_DIR = $(BASE_DIR)rd
VIS_DIR = $(BASE_DIR)vis

INC = 
LIBS =

INC += -I $(BASE_DIR) -I $(BASE_DIR)third-party


#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------


rwildcard = $(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))


RD_DEPS = 	$(call rwildcard, $(RD_DIR),*.hpp)	\
			$(call rwildcard, $(RD_DIR),*.h)	\
			$(call rwildcard, $(RD_DIR),*.inl)
