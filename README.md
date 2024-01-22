# Calling Pytorch models from Java

## Contents

- `JavaTorch.java`: A Java Class that calls a function in `JavaTorch.cpp` through the Java Native Interface (JNI)
- `JavaTorch.cpp`: A JNI compatible function  that loads a previously traced torch model, performs inference and returns the predicted class
- `model.py`: A script that ann generate or evaluate a simple torch model

## Running the example

Download and unzip torchscript 2.1.2, e.g. the CPU version:

    wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.2%2Bcpu.zip
    unzip libtorch-cxx11-abi-shared-with-deps-2.1.2%2Bcpu.zip

Compile with

    make

This will:

- Generate the `JavaTorch.h` header based on the class defined in `JavaTorch.java`
- Compile `JavaTorch.java` into `JavaTorch.class`
- Compile `JavaTorch.cpp` and link with Torchscript

Run: 

    LD_LIBRARY_PATH=libtorch/lib java -Djava.library.path=. JavaTorch models/traced_dummy.pt


## (Optional) Creating/Running the model in python

Install the python torch library, e.g. with conda/mamba

    conda create -n torch pytorch=2.1.2 cpuonly numpy

Evaluate a model with the same dummy input used in `JavaTorch.cpp`

    python model.py --mode run --model_path models/traced_dummy.pt

Or recreate the pytorch model and trace it

    python model.py --mode create --model_path models/traced_dummy.pt

## TODO/Improvements

- Use cmake instead of custom Makefile, torch provides CMAKE recipes. [JNI is also supported by cmake](https://cmake.org/cmake/help/latest/module/FindJNI.html)
- Generate the dummy data in Java and pass it to the C function. Also, pass the logits to Java instead of only the class
- Trace the preprocessing routines 
- Load the model once instead of every time inference is performed. Profile inference.
- Link against libraries in the torch environment instead of additionally downloading `libtorch`

## Notes

Tested with:

- openjdk 17.0.9 2023-10-17
- g++ (GCC) 13.2.1 20230801
- torchscript 2.1.2 cpu version


