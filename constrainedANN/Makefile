
CXX=g++
CFLAGS = -std=gnu++17 -lgfortran -Wall -O3 -w
INC=-I faiss -I include/
LFLAGS=faiss/build/faiss/libfaiss.a OpenBLAS/libopenblas.a -lpthread -lm -ldl -lgfortran -fopenmp

index: clean_index
	$(CXX) $(INC) $(CFLAGS) src/readfile.cpp \
							src/utils.cpp \
							src/FilterIndexHamming.cpp \
							src/index.cpp \
	-o index $(LFLAGS)

query: clean_query
	$(CXX) $(INC) $(CFLAGS) src/readfile.cpp \
							src/utils.cpp \
							src/FilterIndexHamming.cpp \
							src/query.cpp \
	-o query $(LFLAGS)

# --- Python Wrapper Configuration ---
# Detect the python version and flags from the active venv
PY_EXT_SUFFIX := $(shell python3-config --extension-suffix)
PY_INCLUDES   := $(shell python3 -m pybind11 --includes)
PY_CFLAGS     := $(shell python3-config --includes) -fPIC

# The name of the resulting .so file
PY_TARGET     := filter_index_py$(PY_EXT_SUFFIX)

# --- Rules ---

# Rule to compile the Python Wrapper
python_wrapper: clean_py
	$(CXX) $(CFLAGS) -fPIC -shared \
		-DPYBIND11_SIMPLE_GIL_MANAGEMENT \
		$(PY_INCLUDES) \
		src/utils.cpp \
		src/FilterIndexHamming.cpp \
		src/wrapper.cpp \
		-o $(PY_TARGET) \
		-I./include \
		$(LFLAGS)

clean_py:
	rm -f filter_index_py*.so
	
clean_index:
	rm -f index
clean_query:
	rm -f query

.PHONY: clean all

debug: CXXFLAGS += -DDEBUG -g
debug: all

release: CXXFLAGS += -O2
release: all
