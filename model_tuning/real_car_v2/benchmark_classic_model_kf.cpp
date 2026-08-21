#include "cuda_mppi_controller/lateral_velocity_kf.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

int main(int argc,char **argv){
    const int warmup=10000,iterations=argc>1?std::max(1,std::atoi(argv[1])):200000;
    mppi::LateralVelocityKFParams p;mppi::LateralVelocityKF kf;kf.initialize(p);
    auto run=[&](int count){
        for(int i=0;i<count;++i){const float t=.02f*i;
            kf.update(2.f*std::sin(.2f*t),2.f*(1.f-std::cos(.2f*t)),.2f*t,2.f,
                .02f,.4f,.01f,.8f,.08f,2.f);}
    };
    run(warmup);const auto begin=std::chrono::steady_clock::now();run(iterations);
    const auto end=std::chrono::steady_clock::now();
    const double total_us=std::chrono::duration<double,std::micro>(end-begin).count();
    std::cout<<"iterations="<<iterations<<" total_ms="<<total_us/1000.0
             <<" mean_us="<<total_us/iterations
             <<" budget_20ms_percent="<<(total_us/iterations)/200.0<<'\n';
    constexpr int batch_size=100;const int batches=std::max(100,iterations/batch_size);
    std::vector<double> samples;samples.reserve(batches);
    for(int i=0;i<batches;++i){const auto a=std::chrono::steady_clock::now();run(batch_size);
        const auto b=std::chrono::steady_clock::now();samples.push_back(
            std::chrono::duration<double,std::micro>(b-a).count()/batch_size);}
    std::sort(samples.begin(),samples.end());
    const auto percentile=[&](double q){return samples[static_cast<std::size_t>(q*(samples.size()-1))];};
    std::cout<<"batched_update_us p50="<<percentile(.50)<<" p95="<<percentile(.95)
             <<" p99="<<percentile(.99)<<" max="<<samples.back()<<'\n';
}
