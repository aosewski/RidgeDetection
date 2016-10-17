#/***************************************************************************
# * Author: Adam Rogowiec
# *
# ***************************************************************************/

include ../../gpu_common.mk

TARGET = rd

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

DEPS =	$(RD_DEPS)

all: build


build: $(TARGET)
#-------------------------------------------------------------------------------

$(TARGET): $(TARGET).o sim.o 
	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ $^ $(LIBS)

#-------------------------------------------------------------------------------

$(TARGET).o : $(TARGET).cu $(TARGET).cuh $(DEPS)
	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -c $@ $< $(LIBS)

sim.o : sim.cpp $(DEPS)
	$(NVCC) $(DEFINES) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -c $@ $< $(LIBS)

clean:
	rm -f *.o $(TARGET) *.png *.txt

clean-images:
	rm -f *.png
	