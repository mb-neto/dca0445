.SUFFIXES:
.SUFFIXES: .c .cpp

GCC = g++
CC = gcc

.c:
	$(CC) -I$(INCDIR) $(CFLAGS) $< $(GL_LIBS) -o $@

.cpp:
	$(GCC) -Wall -Wunused -std=c++11 -O2 $< -o $@ `pkg-config --cflags --libs opencv4`