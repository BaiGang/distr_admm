
#OPT ?= -O2 -DNDEBUG       # (A) Production use (optimized mode)
OPT ?= -g2  # (B) Debug mode, w/ full line-level debugging symbols
#OPT ?= -O2 -g2 -DNDEBUG # (C) Profiling mode: opt, but w/debugging symbols

CC=gcc
CXX=g++


CFLAGS += -I./ -I./include -I./third_party/root/include -Wall $(OPT) -pthread -fPIC 
CXXFLAGS += -I. -I./include -I./third_party/root/include -Wall -std=c++0x -DDMLC_USE_CXX11 $(OPT) -pthread -fPIC -fopenmp

LDFLAGS += -L./third_party/root/lib -L/usr/local/lib
LIBS += -lpthread -lrabit -ldmlc -lhdfs -llbfgs -lrt

LIBOBJECTS = src/sample_set.o

TESTS = 

TESTOBJECTS = $(TESTS:.cc=.o)

all: program

check: all_test

clean:
	rm -rf $(LIBOBJECTS) $(TESTOBJECTS) all_test admm

lint:
	python cpplint.py src/*.h src/*.cc src/*.cpp

program: $(LIBOBJECTS) src/admm_main.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LIBOBJECTS) src/admm_main.cpp -o admm $(LIBS)

all_test: $(LIBOBJECTS) $(TESTOBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LIBOBJECTS) $(TESTOBJECTS) -o all_test test/gtest-all.cc test/gtest_main.cc $(LIBS)

.cc.o:
	$(CXX) $(CXXFLAGS) -c $< -o $@

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@



