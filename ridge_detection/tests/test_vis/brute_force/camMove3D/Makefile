#/***************************************************************************
# * Author: Adam Rogowiec
# *
# ***************************************************************************/

include ../../../../cpu_common.mk

GLFW3_CFLAGS := $(shell pkg-config --cflags glfw3)
GLFW3_LIBS := $(shell pkg-config --static --libs glfw3)


TARGET1 := camMove3D
TARGET2 := camMove3Dv2

INC += -I$(VIS_DIR)

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

DEPS :=	$(RD_DEPS)
		# $(VIS_DIR)	

all: build1 build2
 
build1: $(TARGET1)

build2: $(TARGET2)

#---------- TARGET1 --------------------------------------------------------------

$(TARGET1): $(TARGET1).o vis.o gl3w.o shader.o sb7ktx.o camera.o
	$(CC) $(DEFINES) $(CCFLAGS) $(GLFW3_CFLAGS) $(INC) -o $@ $+ $(LIBS) $(GLFW3_LIBS)

$(TARGET1).o: $(TARGET1).cpp $(DEPS)
	$(CC) $(CCFLAGS) $(INC) -o $@ -c $<

#---------- TARGET2 --------------------------------------------------------------

$(TARGET2): $(TARGET2).o vis.o gl3w.o shader.o sb7ktx.o camera.o
	$(CC) $(DEFINES) $(CCFLAGS) $(GLFW3_CFLAGS) $(INC) -o $@ $+ $(LIBS) $(GLFW3_LIBS)

$(TARGET2).o: $(TARGET2).cpp $(DEPS)
	$(CC) $(CCFLAGS) $(INC) -o $@ -c $<


#---------- OTHER --------------------------------------------------------------

vis.o: $(VIS_DIR)/src/vis.cpp $(VIS_DIR)/inc/vis.h $(VIS_DIR)/inc/vis_ext.h
	$(CC) $(CCFLAGS) $(GLFW3_CFLAGS) $(INC) -o $@ -c $< $(GLFW3_LIBS)

gl3w.o: $(VIS_DIR)/src/gl3w.cpp 
	$(CC) $(CCFLAGS) $(INC) -o $@ -c $<

sb7ktx.o: $(VIS_DIR)/src/sb7ktx.cpp
	$(CC) $(CCFLAGS) $(INC) -o $@ -c $<

shader.o: $(VIS_DIR)/src/shader.cpp
	$(CC) $(CCFLAGS) $(INC) -o $@ -c $<

camera.o: $(VIS_DIR)/src/camera.cpp $(VIS_DIR)/inc/camera.h
	$(CC) $(CCFLAGS) $(INC) -o $@ -c $<

run1: build1
	./$(TARGET1)

run2: build2
	./$(TARGET2)

clean:
	rm -f *.o $(TARGET1) $(TARGET2)