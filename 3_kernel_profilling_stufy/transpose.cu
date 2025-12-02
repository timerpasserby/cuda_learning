#include <bits/stdc++.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <sys/time.h>

#include <ctime>
#include <iostream>
#include <random>

using namespace std;
#define PERF(name) Perf Perf_##name##__COUNTER__(#name)
#define PERF_CPU(name) PerfCPU perf_CPU_##__COUNTER__(#name)

class PerfCPU {
 public:
  PerfCPU(const std::string& name) : m_name(name) {
    gettimeofday(&m_start, NULL);
  }

  ~PerfCPU() {}

 private:
  std::string m_name;
  struct timeval m_start, m_end;
}