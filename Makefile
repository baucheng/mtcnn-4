PROJECT := mtcnn

CUDA_DIR := /usr/local/cuda
CUDA_INCLUDE_DIR := $(CUDA_DIR)/include
PROTO_INCLUDE_DIR := /home/dkk/projects/caffe/build/src
INCLUDE_DIRS := /home/dkk/projects/caffe/include $(CUDA_INCLUDE_DIR) $(PROTO_INCLUDE_DIR)
LIBRARY_DIRS := /home/dkk/projects/caffe/build/lib

LIBRARIES += caffe boost_system stdc++ opencv_core opencv_highgui opencv_imgproc glog
CXX = g++

WARNINGS := -Wall -Wno-sign-compare
COMMON_FLAGS += -O2
CXXFLAGS +=  $(COMMON_FLAGS) $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
LINKFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) \
             $(foreach library,$(LIBRARIES),-l$(library))

test:
	$(CXX) caffe_mtcnn.cpp -o mtcnn.bin $(WARNINGS) $(CXXFLAGS) $(LINKFLAGS)
