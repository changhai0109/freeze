CUDA_HOME = $(shell dirname $(shell which nvcc))/..

NVCC = nvcc -ccbin=$(CXX) -D_FORCE_INLINES
CXX  ?= g++

NVCC_VER_REQ = 10.1
NVCC_VER = $(shell $(NVCC) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
NVCC_VER_CHECK = $(shell echo "$(NVCC_VER) >= $(NVCC_VER_REQ)" | bc)

ifeq ($(NVCC_VER_CHECK),0)
$(error ERROR: nvcc version >= $(NVCC_VER_REQ) required to compile an nvbit tool!)
endif

NVBIT_PATH = ../nvbit_release_x86_64/core
INCLUDES = -I$(NVBIT_PATH) -I. -I$(CUDA_HOME)/include

LIBS = -L$(NVBIT_PATH) -lnvbit
NVCC_PATH = -L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

ARCH ?= all

# ---- sources ----
SRC_DIRS := . handlers
CC_SRC := $(foreach d,$(SRC_DIRS),$(wildcard $(d)/*.cc))
CU_SRC := $(foreach d,$(SRC_DIRS),$(wildcard $(d)/*.cu))

CU_OBJ  = $(CU_SRC:.cu=.o)
CC_OBJ  = $(CC_SRC:.cc=.o)

OBJECTS = $(CU_OBJ) $(CC_OBJ)

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))
NVBIT_TOOL  := $(current_dir).so

# ---- targets ----
all: $(NVBIT_TOOL)

$(NVBIT_TOOL): $(OBJECTS) $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 $(OBJECTS) \
	    $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static \
	    -shared -o $@

# ---- compilation rules ----
%.o: %.cu
	$(NVCC) -dc -c -std=c++11 $(INCLUDES) \
	    -Xptxas -cloning=no \
	    -Xcompiler -Wall -Xcompiler -fPIC \
	    -arch=$(ARCH) -O3 $< -o $@

%.o: %.cc
	$(CXX) -std=c++11 -Wall -fPIC $(INCLUDES) -O3 \
	    -c $< -o $@

clean:
	rm -f *.so *.o
