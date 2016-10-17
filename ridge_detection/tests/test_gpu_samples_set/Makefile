#/***************************************************************************
# * Author: Adam Rogowiec
# *
# ***************************************************************************/

include ../../gpu_common.mk

TARGET = test_gpu_samples_set
GENERATOR = samples_generator


#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

DEPS =		$(RD_DEPS) \
			$(BASE_DIR)tests/test_gpu_samples_set/ut_gpu_samples_set.cuh


all: build

build: $(TARGET)

# $(TARGET).o: $(TARGET).cu
# 	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ -c $^

# $(TARGET): $(TARGET).o
# 	$(NVCC) $(SM_TARGETS) $(CPU_ARCH) $(NVCCFLAGS) -o $@ $+ $(LIBS)

$(TARGET): $(TARGET).cu $(DEPS)
	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ $< $(LIBS)

$(GENERATOR): $(GENERATOR).cu $(DEPS)
	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ $< $(LIBS)

run: build
	./$(TARGET)

clean:
	rm -f *.o $(TARGET) *.png
	
clobber: 
	clean