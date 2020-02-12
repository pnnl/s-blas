 # =====================================================================================
 #
 #       Filename:  shared.mk
 #
 #    Description:  This file define the parameters for compilation & running.
 #
 #        Version:  0.1
 #        Created:  09/25/2019 16:15:18 PM
 #       Revision:  none
 #       Compiler:  make
 #
 #         Author:  Ang Li, Jieyang Chen, Chenhao Xie, Jiajia Li, Jesun Firoz
 #        Company:  Pacific Northwest National Laboratory
 #
 # =====================================================================================

#using bash for shell
SHELL = /bin/bash

# GPU architecture compute capability
ARCH = -gencode=arch=compute_70,code=compute_70

#NVSHMEM_HOME ?= /home/local/nvshmem/
NVSHMEM_HOME ?= /home/lian599/raid/nvshmem/nvshmem_0.3.0/build
#NVSHMEM_HOME ?= /qfs/people/lian599/local/hpda_final/nvshmem_0.3.0/build/


# CUDA environment parameters
#CUDA_PATH = /autofs/nccs-svm1_sw/summit/cuda/10.1.168/
CUDA_PATH = /usr/local/cuda/
#CUDA_PATH = /share/apps/cuda/10.1.243/
#CUDA_PATH = /share/apps/cuda/10.1/

# CUDA sample path
CUDA_SDK_PATH = $(CUDA_PATH)/samples

# CUDA libraries
LIB_DIR = $(CUDA_PATH)/lib64

# Host compiler
CC = gcc

# Host compiler flags
CC_FLAGS = -O3 

# CUDA compiler
NVCC = $(CUDA_PATH)/bin/nvcc

# CUDA compiler flags
NVCC_FLAGS = $(ARCH) -O3 -w -m64 --default-stream per-thread

# CUDA include library
NVCC_INCLUDE = -I$(CUDA_PATH)/include -I./common -I$(CUDA_SDK_PATH)/common/inc -I../include

# CUDA lib
NVCC_LIB = -lcuda -lcudart -lcusparse -Xcompiler -fopenmp   # -lmpich -lmpl -lnccl

# CUDA lib path
NVCC_LIB_PATH = -L. -L$(CUDA_PATH)/lib64/ -L/usr/lib/ -L/usr/lib64/

# Linking 
LINK_FLAG = $(NVCC_INCLUDE) $(NVCC_LIB_PATH) $(NVCC_LIB) -lstdc++ -lm

# value type
VALUE_TYPE = double
