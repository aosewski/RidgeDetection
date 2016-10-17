#/***************************************************************************
# * Author: Adam Rogowiec
# *
# ***************************************************************************/

GLFW3_CFLAGS = $(shell pkg-config --cflags glfw3)
GLFW3_LIBS = $(shell pkg-config --static --libs glfw3)

CC	 		:= /usr/bin/g++-4.9
# CC	 		:= g++
CCFLAGS 	+= -std=c++11 -O2 -Wall

TARGET = glfw_test

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

all: glfwTest
 
glfwTest: $(TARGET)

$(TARGET): $(TARGET).cpp 
	$(CC) $(CCFLAGS) $(GLFW3_CFLAGS) -o $@ $< $(GLFW3_LIBS)

run: glfwTest
	./$(TARGET)

clean:
	rm -f *.o $(TARGET)