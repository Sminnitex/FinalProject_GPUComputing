CC = nvcc
LIB_HOME = $(CURDIR)
LIBS = -L$(LIB_HOME)/lib64 -lcusparse -lcudart
INCLUDE = -Isrc
OPT = -std=c++14 -O0

#CODE
MAIN = sparse.cu

######################################################################################################
BUILDDIR := obj
TARGETDIR := bin

all: $(TARGETDIR)/final

debug: OPT += -DDEBUG -g
debug: NVCC_FLAG += -G
debug: all

$(TARGETDIR)/final: $(MAIN) $(OBJECTS)
	@mkdir -p $(@D)
	$(CC) $^ --gpu-architecture=sm_50  -o $@ $(INCLUDE) $(LIBS) $(OPT) 

clean:
	rm  $(TARGETDIR)/*
#	rm $(TARGETDIR) $(BUILDDIR)/*.o