#/******************************************************************************
# * Copyright (c) 2011, Duane Merrill.  All rights reserved.
# * Copyright (c) 2011-2015, NVIDIA CORPORATION.  All rights reserved.
# * 
# * Redistribution and use in source and binary forms, with or without
# * modification, are permitted provided that the following conditions are met:
# *	 * Redistributions of source code must retain the above copyright
# *	   notice, this list of conditions and the following disclaimer.
# *	 * Redistributions in binary form must reproduce the above copyright
# *	   notice, this list of conditions and the following disclaimer in the
# *	   documentation and/or other materials provided with the distribution.
# *	 * Neither the name of the NVIDIA CORPORATION nor the
# *	   names of its contributors may be used to endorse or promote products
# *	   derived from this software without specific prior written permission.
# * 
# * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
# * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *
#******************************************************************************/

#/***************************************************************************
# * Author: Adam Rogowiec
# *	Modified for use in Master diploma project.
# *
# ***************************************************************************/

#-------------------------------------------------------------------------------
# Commandline Options
#-------------------------------------------------------------------------------

# [sm=<XXX,...>] Compute-capability to compile for, e.g., "sm=200,300,350" (SM35 by default).
  
COMMA = ,
ifdef sm
	SM_ARCH = $(subst $(COMMA),-,$(sm))
else 
    SM_ARCH = 350
endif

##
## -gencode=arch=compute_52,code=\"sm_52,compute_52\" 
##  means:
## -gencode=arch=compute_52 	- compile code for the vritual architecture 52,
## -code=\"sm_52,compute_52\"	- generate binary code for real sm_52 architecture and include
## 								  ptx code for virtual arch 52, for JIT compilation on
## 								  higer architectures
##
ifeq (520, $(findstring 520, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_52,code=\"sm_52,compute_52\" 
    SM_DEF 		+= -DSM520
    TEST_ARCH 	= 520
endif
ifeq (500, $(findstring 500, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_50,code=\"sm_50,compute_50\" 
    SM_DEF 		+= -DSM500
    TEST_ARCH 	= 500
endif
ifeq (370, $(findstring 370, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_35,code=\"sm_37,compute_37\" 
    SM_DEF 		+= -DSM370
    TEST_ARCH 	= 370
endif
ifeq (350, $(findstring 350, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_35,code=\"sm_35,compute_35\" 
    SM_DEF 		+= -DSM350
    TEST_ARCH 	= 350
endif
ifeq (300, $(findstring 300, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_30,code=\"sm_30,compute_30\"
    SM_DEF 		+= -DSM300
    TEST_ARCH 	= 300
endif
ifeq (210, $(findstring 210, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_20,code=\"sm_21,compute_20\"
    SM_DEF 		+= -DSM210
    TEST_ARCH 	= 210
endif
ifeq (200, $(findstring 200, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_20,code=\"sm_20,compute_20\"
    SM_DEF 		+= -DSM200
    TEST_ARCH 	= 200
endif


DEFINES += $(SM_DEF)

# [cdp=<0|1>] CDP enable option (default: no)	---> CUDA Dynamic Parallelism
ifeq ($(cdp), 1)
	DEFINES += -DCUB_CDP
    NVCCFLAGS += -rdc=true -lcudadevrt
    CDP_SUFFIX += cdp
else
	CDP_SUFFIX += nocdp
endif


# [force32=<0|1>] Device addressing mode option (64-bit device pointers by default) 
ifeq ($(force32), 1)
	CPU_ARCH = -m32
	CPU_ARCH_SUFFIX = i386
else
	CPU_ARCH = -m64
	CPU_ARCH_SUFFIX = x86_64
endif


# [abi=<0|1>] CUDA ABI option (enabled by default) 
ifneq ($(abi), 0)
	ABI_SUFFIX = abi
else 
	NVCCFLAGS += -Xptxas -abi=no
	ABI_SUFFIX = noabi
endif


# [open64=<0|1>] Middle-end compiler option (nvvm by default)
ifeq ($(open64), 1)
	NVCCFLAGS += -open64
	PTX_SUFFIX = open64
else 
	PTX_SUFFIX = nvvm
endif


# [verbose=<0|1|2>] Verbose toolchain output from nvcc option
# and verbose kernel properties (regs, smem, cmem, etc.);
ifeq ($(verbose), 1)
	NVCCFLAGS += -v
else ifeq ($(verbose), 2)
	# NVCCFLAGS += -v -Xptxas -v
	NVCCFLAGS += -v --resource-usage
endif


# [keep=<0|1>] Keep intermediate compilation artifacts option
# passing additionally --clean-targets or -clean will clean all files that would be otherwise created by given command
ifeq ($(keep), 1)
	NVCCFLAGS += -keep
endif

# [debug=<0|1|2>] Generate debug mode code
ifeq ($(debug), 1)
	# -g - host debug informations
	# -g3 - host debug inf level 3(max)
	# -0g - host debug optimizations (recommended by gcc documentation)
	NVCCFLAGS += -G -g -Xcompiler -rdynamic -Xptxas -g -Xcompiler -g3 -Xcompiler -Og
	DEFINES += -DRD_DEBUG
	BUILD_SUFFIX = dbg1
else ifeq ($(debug), 2)
	NVCCFLAGS += -g -lineinfo -Xcompiler -rdynamic -Xptxas -g -Xcompiler -g3 -Xcompiler -Og
	DEFINES += -DRD_DEBUG
	BUILD_SUFFIX = dbg2
else
	NVCCFLAGS += -O3	
	BUILD_SUFFIX = release
endif

# [omp=<0|1>] use openmp
ifeq ($(omp), 1)
	NVCCFLAGS += -Xcompiler -fopenmp
	DEFINES += -DRD_USE_OPENMP
	OPENMP_SUFFIX = omp
else
	OPENMP_SUFFIX = noomp
endif

# [profile=<0|1>] turn on profiling
ifeq ($(profile), 1)
	DEFINES += -DRD_PROFILE
	NVCCFLAGS += -lineinfo
	LIBS += -lnvToolsExt
	BUILD_SUFFIX = prof
endif

ifeq ($(cubin), 1)
	NVCCFLAGS += -Xptxas -preserve-relocs --nvlink-options -preserve-relocs -lineinfo
endif
#-------------------------------------------------------------------------------
# Compiler and compilation platform
#-------------------------------------------------------------------------------

NVCC = "$(shell which nvcc)"
ifdef nvccver
    NVCC_VERSION = $(nvccver)
else
    NVCC_VERSION = $(strip $(shell nvcc --version | grep release | sed 's/.*release //' |  sed 's/,.*//'))
endif

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])

# Default flags: warnings; runtimes for compilation phases
NVCCFLAGS += -Xcompiler -Wall -Xcompiler -Wextra -Xcudafe -\# 

ifeq (WIN_NT, $(findstring WIN_NT, $(OSUPPER)))
    # For MSVC
    # Disable excess x86 floating point precision that can lead to results being labeled incorrectly
    NVCCFLAGS += -Xcompiler /fp:strict
    # Help the compiler/linker work with huge numbers of kernels on Windows
	NVCCFLAGS += -Xcompiler /bigobj -Xcompiler /Zm500
	CC = cl
	ifneq ($(force32), 1)
		CUDART_CYG = "$(shell dirname $(NVCC))/../lib/Win32/cudart.lib"
	else
		CUDART_CYG = "$(shell dirname $(NVCC))/../lib/x64/cudart.lib"
	endif
	CUDART = "$(shell cygpath -w $(CUDART_CYG))"
else
    # For g++
    # Disable excess x86 floating point precision that can lead to results being labeled incorrectly
    NVCCFLAGS += -Xcompiler -ffloat-store
    CC = g++
    
    ifeq ($(NVCC_VERSION), 7.0)
        NVCCFLAGS += -ccbin=/usr/bin/g++-4.9 -std=c++11
    endif
    ifeq ($(NVCC_VERSION), 7.5)
        # --expt-relaxed-constexpr
    	# however above flag is useful when working with c++ numeric_limits, it causes erros when
    	# compiling with debug flags
        NVCCFLAGS += -ccbin=/usr/bin/g++-4.9 -std=c++11
    endif
    ifeq ($(NVCC_VERSION), 8.0)
        NVCCFLAGS += -ccbin=/usr/bin/g++-4.9 -std=c++11
    endif
    # TODO: version checking!
    # ifeq ($(shell test $(NVCC_VERSION) -ge 7; echo $$?),0)
    #     NVCCFLAGS += -ccbin=/usr/bin/g++-4.9 -std=c++11
    # endif

	ifneq ($(force32), 1)
	    CUDART = "$(shell dirname $(NVCC))/../lib/libcudart_static.a"
	else
	    CUDART = "$(shell dirname $(NVCC))/../lib64/libcudart_static.a"
	endif
endif

# Suffix to append to each binary
BIN_SUFFIX = sm$(SM_ARCH)_$(NVCC_VERSION)_$(CDP_SUFFIX)_$(CPU_ARCH_SUFFIX)_$(OPENMP_SUFFIX)_$(BUILD_SUFFIX)

#-------------------------------------------------------------------------------
# Include/library directories and libraries variables
#-------------------------------------------------------------------------------

BASE_DIR = $(dir $(lastword $(MAKEFILE_LIST)))
INC = 
# LIBS =


INC += -I /usr/local/cuda/samples/common/inc/
INC += -I $(BASE_DIR) -I $(BASE_DIR)third-party


#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

RD_DIR = $(BASE_DIR)rd
CUB_DIR = $(BASE_DIR)third-party/cub
VIS_DIR = $(BASE_DIR)vis

rwildcard = $(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))

RD_DEPS =  $(call rwildcard, $(RD_DIR) $(CUB_DIR),*.cuh)	\
			$(call rwildcard, $(RD_DIR) $(CUB_DIR),*.hpp)	\
			$(call rwildcard, $(RD_DIR) $(CUB_DIR),*.h	)	\
			$(call rwildcard, $(RD_DIR) $(CUB_DIR),*.inl)	

