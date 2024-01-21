#include <iostream>
#include <memory>
#include <jni.h>
#include "JavaTorch.h"
#include <torch/script.h>

using namespace std;

torch::List<torch::Tensor> create_dummy_data(int M)
{
  torch::List<torch::Tensor> light_curve;
  light_curve.push_back(torch::ones({1, 1, M})); //TIME
  light_curve.push_back(torch::ones({1, 1, M})); //FLUX
  light_curve.push_back(torch::ones({1, 1, M})); //ERROR 
  light_curve.push_back(torch::ones({1, 1, M}).to(torch::kBool)); //MASK
  return light_curve;
}


JNIEXPORT jint JNICALL Java_JavaTorch_inference(JNIEnv *env, jobject obj, jstring path){

  // Load model
  jboolean isCopy;
  const char *model_path = (env)->GetStringUTFChars(path, &isCopy);
  torch::jit::script::Module module;
  module = torch::jit::load(model_path);
  // Create dummy data 
  std::vector<torch::jit::IValue> inputs;
  auto batch = torch::Dict<std::string, torch::List<torch::Tensor>>();
  batch.insert("light_curve", create_dummy_data(100));
  inputs.push_back(batch);
  //Perform inference
  auto output = module.forward(inputs).toGenericDict();
  std::cout <<  output.at("logits") << '\n';
  return output.at("class").toTensor().item().toInt(); 
}

