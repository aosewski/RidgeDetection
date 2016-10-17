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
include ../../../gpu_common.mk

TARGET = test_inner_decimate

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

DEPS =              $(RD_DEPS)


all: build

build: $(TARGET)

$(TARGET).o: $(TARGET).cu $(DEPS)
	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ -c $<

$(TARGET): $(TARGET).o
	$(NVCC) $(SM_TARGETS) $(CPU_ARCH) $(NVCCFLAGS) -o $@ $+ $(LIBS)

run-big: build
	./$(TARGET) --np=50000 --r1=65 --r2=65 --a=30 --b=75 --s=50

run-small: build
	./$(TARGET) --np=8192 --r1=4 --r2=4 --a=12 --b=5 --s=3
clean:
	rm -f *.o $(TARGET) *.png

clobber:
	clean