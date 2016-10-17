#/***************************************************************************
# * Author: Adam Rogowiec
# *
# ***************************************************************************/

include ../../cpu_common.mk

TARGET := test_cpu_samples_set
GENERATOR = samples_generator
#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

DEPS =		$(RD_DEPS) \
			$(BASE_DIR)tests/test_cpu_samples_set/ut_cpu_samples_set.hpp


all: build

build: $(TARGET)

# $(TARGET).o: $(TARGET).cpp 
# 	$(CC) $(DEFINES) $(CCFLAGS) $(INC) -o $@ -c $<

# $(TARGET): $(TARGET).o
# 	$(CC) $(LDFLAGS) -o $@ $+ $(LIBS)

$(TARGET): $(TARGET).cpp $(DEPS)
	$(CC) $(DEFINES) $(CCFLAGS) $(INC) -o $@ $< $(LIBS)

$(GENERATOR): $(GENERATOR).cpp $(DEPS)
	$(CC) $(DEFINES) $(CCFLAGS) $(INC) -o $@ $< $(LIBS)
	
run: build
	./$(TARGET)

clean:
	rm -f *.o $(TARGET) *.png

