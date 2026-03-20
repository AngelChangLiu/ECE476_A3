
EXECUTABLE := cudaSaxpy

CU_FILES   := saxpy.cu

CU_DEPS    :=

CC_FILES   := main.cpp

###########################################################
ARCH := $(shell uname -m)

ifeq ($(ARCH), aarch64)
    CXX = g++
else
    CXX = g++ -m64
endif

OBJDIR=objs
CXXFLAGS=-O3 -Wall
ifeq ($(ARCH), Darwin)
# Building on mac
LDFLAGS=-L/usr/local/depot/cuda-8.0/lib/ -lcudart
else
# Building on Linux
LDFLAGS=-lcudart
endif
NVCC=nvcc
NVCCFLAGS=-O3 -m64 -Wno-deprecated-gpu-targets


OBJS=$(OBJDIR)/main.o  $(OBJDIR)/saxpy.o


.PHONY: clean run

default: $(EXECUTABLE)

$(OBJDIR):
	mkdir -p $(OBJDIR)/

clean:
	rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)

$(EXECUTABLE): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp | $(OBJDIR)
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu | $(OBJDIR)
	$(NVCC) $< $(NVCCFLAGS) -c -o $@

run: $(EXECUTABLE)
	@bash -lc '. ../setup.sh -q && ece476-srun ./$(EXECUTABLE) $(ARGS)'
