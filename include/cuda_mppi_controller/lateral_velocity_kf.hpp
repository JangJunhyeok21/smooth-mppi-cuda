#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace mppi {

struct LateralVelocityKFParams {
    float dt{0.02f};
    float low_speed_threshold{0.5f};
    float max_abs_vy{2.0f};
    // State: [map x, map y, yaw, body vx, body vy, IMU ax bias, IMU ay bias].
    std::array<float,7> process_var{{2e-5f,2e-5f,2e-5f,2e-3f,8e-3f,2e-6f,2e-6f}};
    std::array<float,4> measurement_var{{4e-3f,4e-3f,8e-3f,3e-2f}};
    std::array<float,7> initial_var{{2e-2f,2e-2f,2e-2f,1e-1f,2.5e-1f,1e-1f,1e-1f}};
};

// MCL + wheel-odom + IMU EKF. IMU ax/ay/yaw-rate are causal inputs and
// [MCL x, MCL y, MCL yaw, odom vx] are measurements. No tire model is used.
class LateralVelocityKF {
public:
    static constexpr int N=7;

    void initialize(const LateralVelocityKFParams &params) {
        params_=params;params_.dt=std::max(params_.dt,1e-4f);
        params_.low_speed_threshold=std::max(params_.low_speed_threshold,0.f);
        params_.max_abs_vy=std::max(params_.max_abs_vy,.05f);
        initialized_=false;
    }

    void reset(float x,float y,float yaw,float vx) {
        state_={{x,y,wrap(yaw),vx,0.f,0.f,0.f}};P_.fill(0.f);
        for(int i=0;i<N;++i)P_[i*N+i]=std::max(params_.initial_var[i],1e-9f);
        yaw_rate_=0.f;initialized_=true;
    }

    float update(float measured_x,float measured_y,float measured_yaw,
                 float measured_vx,float imu_yaw_rate,float imu_ax,float imu_ay) {
        if(!initialized_)reset(measured_x,measured_y,measured_yaw,measured_vx);
        if(!finite(measured_x)||!finite(measured_y)||!finite(measured_yaw)||
           !finite(measured_vx))return state_[4];
        const float r=finite(imu_yaw_rate)?imu_yaw_rate:yaw_rate_;
        const float ax=finite(imu_ax)?imu_ax:0.f;
        const float ay=finite(imu_ay)?imu_ay:0.f;
        yaw_rate_=r;
        const float yaw=state_[2],vx=state_[3],vy=state_[4];
        const float bax=state_[5],bay=state_[6],dt=params_.dt;

        std::array<float,N*N> F{};
        for(int i=0;i<N;++i)F[i*N+i]=1.f;
        F[0*N+2]+=dt*(-vx*std::sin(yaw)-vy*std::cos(yaw));
        F[0*N+3]+=dt*std::cos(yaw);F[0*N+4]-=dt*std::sin(yaw);
        F[1*N+2]+=dt*(vx*std::cos(yaw)-vy*std::sin(yaw));
        F[1*N+3]+=dt*std::sin(yaw);F[1*N+4]+=dt*std::cos(yaw);
        F[3*N+4]+=dt*r;F[3*N+5]-=dt;
        F[4*N+3]-=dt*r;F[4*N+6]-=dt;

        state_[0]+=dt*(vx*std::cos(yaw)-vy*std::sin(yaw));
        state_[1]+=dt*(vx*std::sin(yaw)+vy*std::cos(yaw));
        state_[2]=wrap(yaw+dt*r);
        state_[3]+=dt*(ax-bax+r*vy);
        state_[4]+=dt*(ay-bay-r*vx);

        std::array<float,N*N> FP{},predicted{};
        for(int i=0;i<N;++i)for(int j=0;j<N;++j)for(int k=0;k<N;++k)
            FP[i*N+j]+=F[i*N+k]*P_[k*N+j];
        for(int i=0;i<N;++i)for(int j=0;j<N;++j)for(int k=0;k<N;++k)
            predicted[i*N+j]+=FP[i*N+k]*F[j*N+k];
        for(int i=0;i<N;++i)predicted[i*N+i]+=std::max(params_.process_var[i],0.f);
        P_=predicted;

        scalarMeasurement(0,measured_x,params_.measurement_var[0],false);
        scalarMeasurement(1,measured_y,params_.measurement_var[1],false);
        scalarMeasurement(2,measured_yaw,params_.measurement_var[2],true);
        scalarMeasurement(3,measured_vx,params_.measurement_var[3],false);
        if(std::fabs(measured_vx)<params_.low_speed_threshold)state_[4]=0.f;
        state_[4]=std::clamp(state_[4],-params_.max_abs_vy,params_.max_abs_vy);
        return state_[4];
    }

    float getVy()const{return state_[4];}
    float getYawRate()const{return yaw_rate_;}
    float getState(int i)const{return i>=0&&i<N?state_[i]:0.f;}
    float getCovariance(int row,int col)const{return P_[row*N+col];}
    bool isInitialized()const{return initialized_;}

private:
    static bool finite(float v){return std::isfinite(v);}
    static float wrap(float a){
        constexpr float pi=3.14159265358979323846f;
        while(a>pi)a-=2.f*pi;
        while(a<-pi)a+=2.f*pi;
        return a;
    }
    void scalarMeasurement(int index,float measurement,float variance,bool angle) {
        const float innovation=angle?wrap(measurement-state_[index]):measurement-state_[index];
        const float denom=P_[index*N+index]+std::max(variance,1e-9f);
        std::array<float,N> gain{};for(int i=0;i<N;++i)gain[i]=P_[i*N+index]/denom;
        const auto old=P_;
        for(int i=0;i<N;++i)state_[i]+=gain[i]*innovation;
        state_[2]=wrap(state_[2]);
        for(int i=0;i<N;++i)for(int j=0;j<N;++j)
            P_[i*N+j]=old[i*N+j]-gain[i]*old[index*N+j];
        for(int i=0;i<N;++i)for(int j=i+1;j<N;++j){
            const float symmetric=.5f*(P_[i*N+j]+P_[j*N+i]);P_[i*N+j]=P_[j*N+i]=symmetric;
        }
        for(int i=0;i<N;++i)P_[i*N+i]=std::max(P_[i*N+i],1e-9f);
    }
    LateralVelocityKFParams params_{};
    std::array<float,N> state_{};
    std::array<float,N*N> P_{};
    float yaw_rate_{0.f};bool initialized_{false};
};

} // namespace mppi
