JAVA_PATH = ${JAVA_HOME}
CXX = g++
#LIBTORCH_PATH := $(shell python -c 'import torch;print(torch.__path__[0])')
# ASSUMES THAT libtorch was downloaded in this folder:
LIBTORCH_PATH = ./libtorch

all : libJavaTorch.so JavaTorch.class

# $@ matches the target, $< matches the first dependancy
libJavaTorch.so : JavaTorch.o
	${CXX} -shared -fPIC -L${LIBTORCH_PATH}/lib -ltorch -o $@ $<

JavaTorch.o : JavaTorch.cpp JavaTorch.h
	${CXX} -fPIC -I${JAVA_PATH}/include -I${JAVA_PATH}/include/linux -I${LIBTORCH_PATH}/include -c $< -o $@

JavaTorch.h: JavaTorch.java
	${JAVA_PATH}/bin/javac -h . $<

JavaTorch.class: JavaTorch.java
	${JAVA_PATH}/bin/javac $<

test_dummy: libJavaTorch.so 
	python model.py --mode create --model_path models/traced_dummy.pt
	python model.py --mode run --model_path models/traced_dummy.pt
	LD_LIBRARY_PATH=${LIBTORCH_PATH}/lib java -Djava.library.path=./ JavaTorch models/traced_dummy.pt

clean:
	rm libJavaTorch.so JavaTorch.o JavaTorch.h JavaTorch.class


