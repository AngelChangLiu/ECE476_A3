
EXECUTABLE := render

USE_OPENGL ?= auto
NVCC ?= nvcc

CU_FILES   := cudaRenderer.cu benchmark.cu

CU_DEPS    :=

CC_FILES   := main.cpp refRenderer.cpp \
              noise.cpp ppm.cpp sceneLoader.cpp

OPENGL_AVAILABLE := 0
ifneq ($(wildcard /usr/include/GL/glut.h),)
OPENGL_AVAILABLE := 1
endif
ifneq ($(wildcard /usr/include/GL/freeglut.h),)
OPENGL_AVAILABLE := 1
endif
ifeq ($(USE_OPENGL),auto)
USE_OPENGL := $(OPENGL_AVAILABLE)
endif

ifeq ($(USE_OPENGL),1)
CC_FILES += display.cpp
endif

LOGS	   := logs

###########################################################

ARCH := $(shell uname -m)

ifeq ($(ARCH), aarch64)
    CXX = g++
else
    CXX = g++ -m64
endif

OBJDIR=objs
CXXFLAGS=-O3 -Wall -g
HOSTNAME=$(shell hostname)

LIBS       :=
FRAMEWORKS :=

NVCCFLAGS=-O3 -m64 -Wno-deprecated-gpu-targets
LIBS += cudart

ifeq ($(USE_OPENGL),1)
LIBS += GL glut
CXXFLAGS += -DRENDER_USE_OPENGL
endif

CUDA_HOME ?=
ifeq ($(CUDA_HOME),)
NVCC_PATH := $(shell command -v $(NVCC) 2>/dev/null)
ifneq ($(NVCC_PATH),)
CUDA_HOME := $(abspath $(dir $(NVCC_PATH))/..)
else
CUDA_HOME := /usr/local/cuda
endif
endif

LDFLAGS += -L$(CUDA_HOME)/lib64

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

OBJS := $(addprefix $(OBJDIR)/,$(CC_FILES:.cpp=.o) $(CU_FILES:.cu=.o))

.PHONY: clean run_ref run run_cuda run_cpu check check_super

default: $(EXECUTABLE)

$(OBJDIR):
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE) $(LOGS)


$(EXECUTABLE): $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

$(OBJDIR)/%.o: %.cpp | $(OBJDIR)
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu | $(OBJDIR)
		$(NVCC) $< $(NVCCFLAGS) -c -o $@

run_ref:
	@bash -lc '. ../setup.sh -q && ece476-srun ./render_ref $(ARGS)'

run:	default
	@bash -lc '. ../setup.sh -q && ece476-srun ./$(EXECUTABLE) $(ARGS)'

run_cuda:	default
	@bash -lc '. ../setup.sh -q && ece476-srun ./$(EXECUTABLE) -r cuda $(ARGS)'

run_cpu:	default
	@bash -lc '. ../setup.sh -q && ece476-srun ./$(EXECUTABLE) -r cpuref $(ARGS)'

check:	default
	@bash -lc '. ../setup.sh -q && ece476-srun python3 ./checker.py'

check_super:	default
	@bash -lc '. ../setup.sh -q && ece476-srun python3 ./checker.py --super'
