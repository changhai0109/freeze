PROFILE ?= debug
ifeq ($(PROFILE),debug)
  CXXFLAGS = -g -O0 -DDEBUG
  NVCCFLAGS = -g -O0 -DDEBUG
else ifeq ($(PROFILE),release)
  CXXFLAGS = -O3
  NVCCFLAGS = -O3
else
  $(error ERROR: Unknown PROFILE '$(PROFILE)'. Supported: debug, release)
endif

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

# ---- Object Directory ----
# Object files will be placed in a profile-specific build directory
OBJ_DIR := build_$(PROFILE)

# ---- sources ----
SRC_DIRS := . handlers utils tracker
CC_SRC := $(foreach d,$(SRC_DIRS),$(wildcard $(d)/*.cc))
CU_SRC := $(foreach d,$(SRC_DIRS),$(wildcard $(d)/*.cu))

# Object files will be placed in OBJ_DIR, maintaining their original subdirectory structure
CU_OBJ  = $(patsubst %.cu,$(OBJ_DIR)/%.o,$(CU_SRC))
CC_OBJ  = $(patsubst %.cc,$(OBJ_DIR)/%.o,$(CC_SRC))

OBJECTS = $(CU_OBJ) $(CC_OBJ)

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))
NVBIT_TOOL  := $(current_dir).so

# ---- targets ----
all: $(NVBIT_TOOL)
.PHONY: all clean

# Rule to create all necessary object directories
$(OBJ_DIR):
	@mkdir -p $(sort $(dir $(OBJECTS)))

$(NVBIT_TOOL): $(OBJECTS) $(NVBIT_PATH)/libnvbit.a
	$(NVCC) ${NVCCFLAGS} $(OBJECTS) \
	    $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static \
	    -shared -o $@

# ---- compilation rules ----
# CUDA C++ compilation rule
$(OBJ_DIR)/%.o: %.cu | $(OBJ_DIR)
	$(NVCC) -dc -c -std=c++17 $(INCLUDES) \
	    -Xptxas -cloning=no \
	    -Xcompiler -Wall -Xcompiler -fPIC \
	    -arch=$(ARCH) ${NVCCFLAGS} $< -o $@

# Host C++ compilation rule
$(OBJ_DIR)/%.o: %.cc | $(OBJ_DIR)
	$(CXX) -std=c++17 -Wall -fPIC $(INCLUDES) ${CXXFLAGS} \
	    -c $< -o $@

clean:
	@echo "Cleaning up build artifacts..."
	rm -rf *.so $(OBJ_DIR)
	@echo "Clean complete."
