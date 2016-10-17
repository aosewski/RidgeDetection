#/***************************************************************************
# * Author: Adam Rogowiec
# *
# ***************************************************************************/

include ../../../../cpu_common.mk

GLFW3_CFLAGS = $(shell pkg-config --cflags glfw3)
GLFW3_LIBS = $(shell pkg-config --static --libs glfw3)


TARGET = simpleVis3D

INC += -I$(VIS_DIR)

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

DEPS =	$(RD_DEPS)
		# $(VIS_DIR)	

all: build
 
build: $(TARGET)

$(TARGET): $(TARGET).o vis.o gl3w.o 
	$(CC) $(DEFINES) $(CCFLAGS) $(GLFW3_CFLAGS) $(INC) -o $@ $+ $(LIBS) $(GLFW3_LIBS)

$(TARGET).o: $(TARGET).cpp $(DEPS)
	$(CC) $(CCFLAGS) $(INC) -o $@ -c $<

vis.o: $(VIS_DIR)/vis.cpp $(VIS_DIR)/vis.h $(VIS_DIR)/vis_ext.h
	$(CC) $(CCFLAGS) $(GLFW3_CFLAGS) $(INC) -o $@ -c $< $(GLFW3_LIBS)

gl3w.o: $(VIS_DIR)/gl3w.cpp $(VIS_DIR)/GL/gl3w.h
	$(CC) $(CCFLAGS) $(INC) -o $@ -c $<

run: build
	./$(TARGET)

clean:
	rm -f *.o $(TARGET)