#include "JavaTorch.h"
#include <chrono>
#include <iostream>
#include <jni.h>
#include <memory>
#include <torch/cuda.h>
#include <torch/script.h>

#define DEBUG

using namespace std;

std::tuple<at::Tensor, at::Tensor>
to_tensor(double *jtime, double *jmag, double *jerr, int N, bool use_gpu, bool non_blocking, int batch_size) {
  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  auto time = at::from_blob(jtime, {N}, options).to(torch::kFloat32);
  auto mag = at::from_blob(jmag, {N}, options).to(torch::kFloat32);
  auto err = at::from_blob(jerr, {N}, options).to(torch::kFloat32);
  at::Tensor data = at::stack({time, mag, err}, 0).reshape({1, 3, N});
  at::Tensor mask = at::ones({1, N}).to(torch::kBool);
  if (batch_size > 1){
  	data = data.repeat({batch_size, 1, 1});
  	mask = mask.repeat({batch_size, 1});
  }
  if (use_gpu) {
    data = data.to(torch::kCUDA, non_blocking);
    mask = mask.to(torch::kCUDA, non_blocking);
  }
  return std::tuple<at::Tensor, at::Tensor>{data, mask};
}

std::vector<torch::jit::IValue> form_input(double *jtime, double *jmag,
                                           double *jerr, int N, bool use_gpu, bool non_blocking, int batch_size) {
  auto data = to_tensor(jtime, jmag, jerr, N, use_gpu, non_blocking, batch_size);
  auto batch = torch::Dict<
      std::string,
      torch::Dict<std::string, std::tuple<at::Tensor, at::Tensor>>>();
  auto band =
      torch::Dict<std::string, std::tuple<at::Tensor, at::Tensor>>();
  band.insert("g", data);
  band.insert("bp", data);
  band.insert("rp", data);
  batch.insert("light_curve", band);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(batch);
  return inputs;
}

void benchmark(double *jtime, double *jmag, double *jerr, int N, bool use_gpu, bool non_blocking, torch::jit::script::Module module){
    std::vector<torch::jit::IValue> inputs = form_input(jtime, jmag, jerr, N, use_gpu, non_blocking, 128);
    torch::Tensor output = module.forward(inputs).toGenericDict().at("embedding").toTensor();
    if (use_gpu == JNI_TRUE) {
      output = output.to(torch::kCPU, non_blocking);
    }
    //std::cout << output << "\n";
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
#ifdef DEBUG
  auto finish = std::chrono::high_resolution_clock::now();
  std::cout << "Reading data from Java class fields: "
            << std::chrono::duration_cast<milli>(finish - start).count()
            << " ms\n";
  start = std::chrono::high_resolution_clock::now();
#endif
  if (use_gpu == JNI_TRUE)
      torch::cuda::synchronize();

  std::vector<torch::jit::IValue> inputs =
      form_input(time, mag, err, N, use_gpu, true, 256);
  if (use_gpu == JNI_TRUE)
      torch::cuda::synchronize();

#ifdef DEBUG
  finish = std::chrono::high_resolution_clock::now();
  std::cout << "Transforming to tensor and moving to device: "
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
  if (use_gpu == JNI_TRUE)
      torch::cuda::synchronize();
  // Perform inference
  auto output =
      module.forward(inputs).toGenericDict().at("embedding").toTensor();
  if (use_gpu == JNI_TRUE) {
    output = output.to(torch::kCPU, true);
      torch::cuda::synchronize();
  }
#ifdef DEBUG
  finish = std::chrono::high_resolution_clock::now();
  std::cout << "Inference completed in: "
            << std::chrono::duration_cast<milli>(finish - start).count()
            << " ms\n";
#endif
/*
#ifdef DEBUG
  for (int i = 0; i < 10; i++) {
    if (use_gpu == JNI_TRUE)
      torch::cuda::synchronize();
    start = std::chrono::high_resolution_clock::now();
    benchmark(time, mag, err, N, use_gpu, false, module);
    if (use_gpu == JNI_TRUE) {
      torch::cuda::synchronize();
    }
    finish = std::chrono::high_resolution_clock::now();
    std::cout << "Transfer + inference " << i << ": "
              << std::chrono::duration_cast<milli>(finish - start).count()
              << " ms\n";
  }

#endif
*/
  // Return embedding
  std::cout << output.sizes() << "\n";
  int latent_dim = output.sizes()[1];
  jfloatArray embedding = (env)->NewFloatArray(latent_dim);
  env->SetFloatArrayRegion(embedding, 0, latent_dim, output.data_ptr<float>());
  // Clean up
  env->ReleaseDoubleArrayElements(jtime, time, 0);
  env->ReleaseDoubleArrayElements(jmag, mag, 0);
  env->ReleaseDoubleArrayElements(jerr, err, 0);
  return embedding;
}
