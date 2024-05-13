#include "JavaTorch.h"
#include <chrono>
#include <iostream>
#include <jni.h>
#include <memory>
#include <torch/script.h>

#define DEBUG

using namespace std;

std::tuple<torch::Tensor, torch::Tensor>
ArrayToTensor(double *jtime, double *jmag, double *jerr, int N, bool use_gpu) {
  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  auto time = torch::from_blob(jtime, {N}, options).to(torch::kFloat32);
  auto mag = torch::from_blob(jmag, {N}, options).to(torch::kFloat32);
  auto err = torch::from_blob(jerr, {N}, options).to(torch::kFloat32);
  torch::Tensor data = torch::stack({time, mag, err}, 0).reshape({1, 3, N});
  torch::Tensor mask = torch::ones({1, N}).to(torch::kBool);
  if (use_gpu) {
    data = data.to(torch::kCUDA);
    mask = mask.to(torch::kCUDA);
  }
  return std::tuple<torch::Tensor, torch::Tensor>{data, mask};
}

// jdoubleArray *read_field(JNIEnv *env, jclass cls) {
//   jclass jcClass = env->GetObjectClass(cls);
//   jfieldID id = env->GetFieldID(jcClass, "time", "[D");
//   jobject timedata = env->GetObjectField(cls, id);
//   return reinterpret_cast<jdoubleArray *>(&timedata);
// }

JNIEXPORT jfloatArray JNICALL Java_JavaTorch_inference(JNIEnv *env, jobject obj,
                                                       jstring path, jobject lc,
                                                       jboolean use_gpu) {
  c10::InferenceMode guard(true);

// Read data
#ifdef DEBUG
  using milli = std::chrono::milliseconds;
  auto start = std::chrono::high_resolution_clock::now();
#endif

  jboolean isCopy;
  jfieldID id;
  jdouble *time, *mag, *err;
  // jtime = env->GetDoubleArrayElements(time, &isCopy);
  jclass jcClass = env->GetObjectClass(lc);
  id = env->GetFieldID(jcClass, "time", "[D");
  jobject timedata = env->GetObjectField(lc, id);
  jdoubleArray jtime = static_cast<jdoubleArray>(timedata);
  //  jdoubleArray *jtime = read_field(*env, jcClass);
  id = env->GetFieldID(jcClass, "mag", "[D");
  jobject magdata = env->GetObjectField(lc, id);
  jdoubleArray jmag = static_cast<jdoubleArray>(magdata);
  id = env->GetFieldID(jcClass, "err", "[D");
  jobject errdata = env->GetObjectField(lc, id);
  jdoubleArray jerr = static_cast<jdoubleArray>(errdata);
  int N = env->GetArrayLength(jtime);
  // std::cout << (bool)(isCopy==JNI_TRUE) << std::endl;
  time = env->GetDoubleArrayElements(jtime, NULL);
  mag = env->GetDoubleArrayElements(jmag, NULL);
  err = env->GetDoubleArrayElements(jerr, NULL);
  if (time == NULL || mag == NULL || err == NULL) {
    return NULL;
  }
  auto data = ArrayToTensor(time, mag, err, N, use_gpu);
  auto batch = torch::Dict<
      std::string,
      torch::Dict<std::string, std::tuple<torch::Tensor, torch::Tensor>>>();
  auto band =
      torch::Dict<std::string, std::tuple<torch::Tensor, torch::Tensor>>();
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
  if (model_path == NULL) {
    return NULL;
  }
  static torch::jit::script::Module module;
  try {
    module = torch::jit::load(model_path);
  } catch (const c10::Error &e) {
    std::cerr << " Could not load model at: " << model_path << "\n";
    return NULL;
  }
  if (use_gpu == JNI_TRUE) {
    module.to(torch::kCUDA);
  }
  module.eval();
  // torch::set_num_interop_threads(1);

#ifdef DEBUG
  finish = std::chrono::high_resolution_clock::now();
  std::cout << "Loading model took: "
            << std::chrono::duration_cast<milli>(finish - start).count()
            << " ms\n";
  start = std::chrono::high_resolution_clock::now();
#endif

  // Perform inference
  auto output = module.forward(inputs).toGenericDict();
// for (int i = 0; i < 2; i++) {
//   module.forward(inputs).toGenericDict();
// }
#ifdef DEBUG
  finish = std::chrono::high_resolution_clock::now();
  std::cout << "Inference completed in: "
            << std::chrono::duration_cast<milli>(finish - start).count()
            << " ms\n";
#endif
  // Return embedding
  auto emb = output.at("embedding").toTensor().detach().to(torch::kCPU);
  int latent_dim = emb.sizes()[1];
  jfloatArray embedding = (env)->NewFloatArray(latent_dim);
  env->SetFloatArrayRegion(embedding, 0, latent_dim, emb.data_ptr<float>());
  // Clean up
  env->ReleaseDoubleArrayElements(jtime, time, 0);
  env->ReleaseDoubleArrayElements(jmag, mag, 0);
  env->ReleaseDoubleArrayElements(jerr, err, 0);
  return embedding;
}
