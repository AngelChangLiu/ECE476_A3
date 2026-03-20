EXECUTABLE := cudaScan

CU_FILES   := scan.cu

CU_DEPS    :=

CC_FILES   := main.cpp

all: $(EXECUTABLE) $(REFERENCE)

LOGS	   := logs

###########################################################

ARCH := $(shell uname -m)

ifeq ($(ARCH), aarch64)
    CXX = g++
else
    CXX = g++ -m64
endif

OBJDIR=objs
CXXFLAGS=-O3 -Wall
LDFLAGS=-lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -m64 -Wno-deprecated-gpu-targets


OBJS=$(OBJDIR)/main.o  $(OBJDIR)/scan.o


.PHONY: clean run check_scan check_find_repeats

default: $(EXECUTABLE)

$(OBJDIR):
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE) $(LOGS)

$(EXECUTABLE): $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp | $(OBJDIR)
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu | $(OBJDIR)
		$(NVCC) $< $(NVCCFLAGS) -c -o $@

run: default
	@bash -lc '. ../setup.sh -q && ece476-srun ./$(EXECUTABLE) $(ARGS)'

check_scan: default
	@bash -lc '. ../setup.sh -q && ece476-srun python3 ./checker.py scan'

check_find_repeats: default
	@bash -lc '. ../setup.sh -q && ece476-srun python3 ./checker.py find_repeats'
