#include <iostream>
#include <chrono>
#include <memory>
#include <jni.h>
#include <torch/script.h>
#include "JavaTorch.h"

#define DEBUG

using namespace std;


std::tuple<torch::Tensor, torch::Tensor> ArrayToTensor(double *jtime, double *jmag, double *jerr, int N)
{
  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  auto time = torch::from_blob(jtime, {N}, options).to(torch::kFloat32);
  auto mag = torch::from_blob(jmag, {N}, options).to(torch::kFloat32);
  auto err = torch::from_blob(jerr, {N}, options).to(torch::kFloat32);
  torch::Tensor data = torch::stack({time,mag,err}, 0).reshape({1, 3, N});
  torch::Tensor mask = torch::ones({1, N}).to(torch::kBool);
  return std::tuple<torch::Tensor, torch::Tensor>{data, mask};
}


JNIEXPORT jfloatArray JNICALL Java_JavaTorch_inference(JNIEnv *env, jobject obj, jstring path, jdoubleArray time, jdoubleArray mag, jdoubleArray err){
  //Read data
  #ifdef DEBUG
  using milli = std::chrono::milliseconds;
  auto start = std::chrono::high_resolution_clock::now();
  #endif 
  
  jboolean isCopy;
  jdouble *jtime, *jmag, *jerr;
  int N = env->GetArrayLength(time);
  jtime = env->GetDoubleArrayElements(time, &isCopy);
  //std::cout << (bool)(isCopy==JNI_TRUE) << std::endl;
  jmag = env->GetDoubleArrayElements(mag, &isCopy);
  jerr = env->GetDoubleArrayElements(err, &isCopy);
  if (jtime == NULL || jmag == NULL || jerr == NULL){
    return NULL;
  }
  auto data = ArrayToTensor(jtime, jmag, jerr, N);
  auto batch = torch::Dict<std::string, torch::Dict<std::string, 
       std::tuple<torch::Tensor, torch::Tensor>>>();
  auto band = torch::Dict<std::string, std::tuple<torch::Tensor, torch::Tensor>>();
  band.insert("g", data);
  band.insert("bp", data);
  band.insert("rp", data);
  batch.insert("light_curve", band);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(batch);
  
  #ifdef DEBUG
  auto finish = std::chrono::high_resolution_clock::now();
  std::cout << "Transforming data to tensor took: "
            << std::chrono::duration_cast<milli>(finish - start).count()
            << " ms\n";
  start = std::chrono::high_resolution_clock::now();
  #endif
  // Load model
  const char *model_path = (env)->GetStringUTFChars(path, &isCopy);
  if (model_path == NULL){
    return NULL;
  }
  torch::NoGradGuard no_grad;
  static torch::jit::script::Module module;
  try {
    module = torch::jit::load(model_path);
  }
  catch (const c10::Error& e) {
    std::cerr << " Could not load model at: " << model_path << "\n";
    return NULL;
  }
  module.to(torch::kCPU);
  module.eval();
  //torch::set_num_interop_threads(1);
  
  #ifdef DEBUG
  finish = std::chrono::high_resolution_clock::now();
  std::cout << "Loading model took: "
            << std::chrono::duration_cast<milli>(finish - start).count()
            << " ms\n";
  start = std::chrono::high_resolution_clock::now();
  #endif

   // Perform inference
  auto output = module.forward(inputs).toGenericDict();
  //for (int i=0;i<1000;i++){
  //  module.forward(inputs).toGenericDict();
  //}
  #ifdef DEBUG
  finish = std::chrono::high_resolution_clock::now();
  std::cout << "Inference completed in: "
            << std::chrono::duration_cast<milli>(finish - start).count()
            << " ms\n";
  #endif
  // Return embedding
  auto emb = output.at("embedding").toTensor().detach();
  int latent_dim = emb.sizes()[1];
  jfloatArray embedding = (env)->NewFloatArray(latent_dim);
  env->SetFloatArrayRegion(embedding, 0, latent_dim, emb.data_ptr<float>());
  return embedding;
  
}

