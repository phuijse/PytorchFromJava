#include <iostream>
#include <memory>
#include <jni.h>
#include "JavaTorch.h"
#include <torch/script.h>

using namespace std;

std::tuple<torch::Tensor, torch::Tensor> create_dummy_data(int M)
{
  torch::Tensor time = torch::linspace(0, 1000, M);
  torch::Tensor flux = torch::sin(2.0*M_PI*time);
  torch::Tensor ferr = torch::ones({M})*0.1;
  torch::Tensor data = torch::stack({time,flux,ferr}, 0).reshape({1, 3, M});
  torch::Tensor mask = torch::ones({1, M}).to(torch::kBool);
  return std::tuple<torch::Tensor, torch::Tensor>{data, mask};
}


JNIEXPORT jfloatArray JNICALL Java_JavaTorch_inference(JNIEnv *env, jobject obj, jstring path){

  // Load model
  jboolean isCopy;
  const char *model_path = (env)->GetStringUTFChars(path, &isCopy);
  torch::jit::script::Module module;
  module = torch::jit::load(model_path);
  // Create dummy data 
  std::vector<torch::jit::IValue> inputs;
  auto batch = torch::Dict<std::string, torch::Dict<std::string, 
       std::tuple<torch::Tensor, torch::Tensor>>>();
  auto band = torch::Dict<std::string, std::tuple<torch::Tensor, torch::Tensor>>();
  band.insert("g", create_dummy_data(100));
  band.insert("bp", create_dummy_data(100));
  band.insert("rp", create_dummy_data(100));
  batch.insert("light_curve", band);
  inputs.push_back(batch);
  // Perform inference
  auto output = module.forward(inputs).toGenericDict();
  // Return embedding
  auto emb = output.at("embedding").toTensor();
  int latent_dim = emb.sizes()[1];
  jfloatArray embedding = (env)->NewFloatArray(latent_dim);
  env->SetFloatArrayRegion(embedding, 0, latent_dim, emb.data_ptr<float>());
  return embedding;
  
}

