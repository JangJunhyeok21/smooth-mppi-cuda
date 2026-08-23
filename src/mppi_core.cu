#include "cuda_mppi_controller/cuda_mppi_core.hpp"
#include "cuda_mppi_controller/kinematic_residual_weights.hpp"
#include "cuda_mppi_controller/kinematic_mlp_weights.hpp"
#include "cuda_mppi_controller/kinematic_mlp_no_imu_weights.hpp"
#include "cuda_mppi_controller/kinematic_noslip_noimu_direct_weights.hpp"
#include "cuda_mppi_controller/dynamic_imu_recursive_weights.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace mppi
{
    __device__ float rw_enc_w_ih[288*11], rw_enc_w_hh[288*96], rw_enc_b_ih[288], rw_enc_b_hh[288];
    __device__ float rw_dec_w_ih[288*8], rw_dec_w_hh[288*96], rw_dec_b_ih[288], rw_dec_b_hh[288];
    __device__ float rw_head_w[3*96], rw_head_b[3];
    __device__ float rw_feature_mean[11], rw_feature_std[11];
    __device__ float mlp_w1[64*21],mlp_b1[64],mlp_w2[32*64],mlp_b2[32],mlp_w3[3*32],mlp_b3[3];
    __device__ float mlp_feature_mean[11],mlp_feature_std[11];
    __device__ float mlp_noimu_w1[64*18],mlp_noimu_b1[64],mlp_noimu_w2[32*64],mlp_noimu_b2[32],mlp_noimu_w3[3*32],mlp_noimu_b3[3];
    __device__ float mlp_noimu_feature_mean[8],mlp_noimu_feature_std[8];
    __device__ float knd_w1[64*16],knd_b1[64],knd_w2[32*64],knd_b2[32],knd_w3[3*32],knd_b3[3];
    __device__ float knd_mean[16],knd_std[16];
    __device__ float ksd_w1[64*20],ksd_b1[64],ksd_w2[32*64],ksd_b2[32],ksd_w3[3*32],ksd_b3[3];
    __device__ float ksd_mean[20],ksd_std[20];
    __device__ float dir_w1[64*20],dir_b1[64],dir_w2[32*64],dir_b2[32],dir_w3[3*32],dir_b3[3];
    __device__ float dir_mean[20],dir_std[20];
    __device__ float dmr_w1[64*22],dmr_b1[64],dmr_w2[32*64],dmr_b2[32],dmr_w3[3*32],dmr_b3[3];
    __device__ float dmr_mean[22],dmr_std[22];
    __device__ float dmr24_w1[64*24],dmr24_b1[64],dmr24_w2[32*64],dmr24_b2[32],dmr24_w3[3*32],dmr24_b3[3];
    __device__ float dmr24_mean[24],dmr24_std[24];

    __device__ inline float sigmoidf_fast(float x) { return 1.f/(1.f+__expf(-x)); }

    template<int INPUT>
    __device__ void gru_step(const float *input, float *h, const float *wih,
                             const float *whh, const float *bih, const float *bhh) {
        float reset[RESIDUAL_HIDDEN], update[RESIDUAL_HIDDEN], candidate[RESIDUAL_HIDDEN];
        for(int j=0;j<RESIDUAL_HIDDEN;++j) {
            float ir=bih[j], iz=bih[RESIDUAL_HIDDEN+j], in=bih[2*RESIDUAL_HIDDEN+j];
            float hr=bhh[j], hz=bhh[RESIDUAL_HIDDEN+j], hn=bhh[2*RESIDUAL_HIDDEN+j];
            for(int i=0;i<INPUT;++i) {
                ir += wih[j*INPUT+i]*input[i];
                iz += wih[(RESIDUAL_HIDDEN+j)*INPUT+i]*input[i];
                in += wih[(2*RESIDUAL_HIDDEN+j)*INPUT+i]*input[i];
            }
            for(int i=0;i<RESIDUAL_HIDDEN;++i) {
                hr += whh[j*RESIDUAL_HIDDEN+i]*h[i];
                hz += whh[(RESIDUAL_HIDDEN+j)*RESIDUAL_HIDDEN+i]*h[i];
                hn += whh[(2*RESIDUAL_HIDDEN+j)*RESIDUAL_HIDDEN+i]*h[i];
            }
            reset[j]=sigmoidf_fast(ir+hr); update[j]=sigmoidf_fast(iz+hz);
            candidate[j]=tanhf(in+reset[j]*hn);
        }
        for(int j=0;j<RESIDUAL_HIDDEN;++j) h[j]=(1.f-update[j])*candidate[j]+update[j]*h[j];
    }

    __global__ void encode_residual_history(const float *history, float *hidden) {
        if(threadIdx.x||blockIdx.x) return;
        for(int i=0;i<RESIDUAL_HIDDEN;++i) hidden[i]=0.f;
        for(int t=0;t<RESIDUAL_HISTORY;++t)
            gru_step<RESIDUAL_FEATURES>(history+t*RESIDUAL_FEATURES,hidden,
                rw_enc_w_ih,rw_enc_w_hh,rw_enc_b_ih,rw_enc_b_hh);
    }
    __host__ __device__ inline float fast_cos(float x) {
    #ifdef __CUDA_ARCH__
        return __cosf(x); 
    #else
        return cosf(x); 
    #endif
    }

    __host__ __device__ inline float fast_sin(float x) {
    #ifdef __CUDA_ARCH__
        return __sinf(x);
    #else
        return sinf(x);
    #endif
    }

    __host__ __device__ inline float fast_exp(float x) {
    #ifdef __CUDA_ARCH__
        return __expf(x);
    #else
        return expf(x);
    #endif
    }

    __host__ __device__ float angle_normalize(float angle)
    {
        while (angle > M_PI) angle -= 2.0f * M_PI;
        while (angle < -M_PI) angle += 2.0f * M_PI;
        return angle;
    }

    __host__ __device__ State update_kinematic(const State &s, const Control &u, const Params &p)
    {
        const float wheelbase = p.l_f + p.l_r;
        // Identified mapping from /drive steering command to effective tire
        // angle.  The selected checkpoint was trained with the no-slip branch.
        const float steer = fminf(.55f,fmaxf(-.55f,
            p.kinematic_steer_scale*u.steer+p.kinematic_steer_bias));
        const float beta = p.kinematic_no_slip ? 0.f :
            atanf((p.l_r / wheelbase) * tanf(steer));
        const float next_v = fminf(p.max_speed, fmaxf(p.min_speed, s.v + u.accel * p.dt));
        const float yaw_rate = s.v * fast_cos(beta) * tanf(steer) / wheelbase;

        State next_s{};
        next_s.x = s.x + s.v * fast_cos(s.yaw + beta) * p.dt;
        next_s.y = s.y + s.v * fast_sin(s.yaw + beta) * p.dt;
        next_s.yaw = angle_normalize(s.yaw + yaw_rate * p.dt);
        next_s.v = next_v;
        next_s.ax = u.accel;
        next_s.vy = next_v * fast_sin(beta);
        next_s.omega = yaw_rate;
        next_s.ay = next_v * yaw_rate;
        next_s.slip_angle = beta;
        return next_s;
    }

    __device__ State update_kinematic_residual(const State &s, const Control &u,
                                                const Params &p, float *hidden)
    {
        State classic_s=update_kinematic(s,u,p);
        const float speed_cmd=fminf(p.max_speed,fmaxf(p.min_speed,s.v+u.accel*p.dt));
        const int norm_index[8]={0,1,2,3,4,8,9,10};
        float raw[8]={s.v,s.vy,s.omega,u.steer,speed_cmd,
                      classic_s.v,classic_s.vy,classic_s.omega};
        float input[8];
        for(int i=0;i<8;++i) input[i]=(raw[i]-rw_feature_mean[norm_index[i]])/
                                      rw_feature_std[norm_index[i]];
        gru_step<8>(input,hidden,rw_dec_w_ih,rw_dec_w_hh,rw_dec_b_ih,rw_dec_b_hh);
        float correction[3];
        const float scale[3]={8.f,8.f,30.f};
        for(int o=0;o<3;++o){
            float z=rw_head_b[o];
            for(int i=0;i<RESIDUAL_HIDDEN;++i) z+=rw_head_w[o*RESIDUAL_HIDDEN+i]*hidden[i];
            correction[o]=tanhf(z)*scale[o];
        }
        State next_s=classic_s;
        next_s.v=classic_s.v+correction[0]*p.dt;
        next_s.vy=classic_s.vy+correction[1]*p.dt;
        next_s.omega=classic_s.omega+correction[2]*p.dt;
        next_s.x=s.x+(next_s.v*fast_cos(s.yaw)-next_s.vy*fast_sin(s.yaw))*p.dt;
        next_s.y=s.y+(next_s.v*fast_sin(s.yaw)+next_s.vy*fast_cos(s.yaw))*p.dt;
        next_s.yaw=angle_normalize(s.yaw+next_s.omega*p.dt);
        next_s.slip_angle=atan2f(next_s.vy,fabsf(next_s.v)+1e-5f);
        next_s.ay=(next_s.vy-s.vy)/p.dt+next_s.v*next_s.omega;
        return next_s;
    }

    __device__ State update_kinematic_mlp_residual(const State &s,const Control &u,
                                                    const Params &p,float *command_history)
    {
        State classic_s=update_kinematic(s,u,p);
        float speed_cmd=fminf(p.max_speed,fmaxf(p.min_speed,s.v+u.accel*p.dt));
        float raw[21]={s.v,s.vy,s.omega,p.residual_imu[0],p.residual_imu[1],p.residual_imu[2],
                       u.steer,speed_cmd,classic_s.v,classic_s.vy,classic_s.omega};
        for(int i=0;i<10;++i) raw[11+i]=command_history[i];
        const int ni[21]={0,1,2,5,6,7,3,4,8,9,10,3,4,3,4,3,4,3,4,3,4};
        float in[21],h1[64],h2[32];
        for(int i=0;i<21;++i) in[i]=(raw[i]-mlp_feature_mean[ni[i]])/mlp_feature_std[ni[i]];
        for(int o=0;o<64;++o){float z=mlp_b1[o];for(int i=0;i<21;++i)z+=mlp_w1[o*21+i]*in[i];h1[o]=z*sigmoidf_fast(z);}
        for(int o=0;o<32;++o){float z=mlp_b2[o];for(int i=0;i<64;++i)z+=mlp_w2[o*64+i]*h1[i];h2[o]=z*sigmoidf_fast(z);}
        float corr[3],scale[3]={8.f,8.f,30.f};
        for(int o=0;o<3;++o){float z=mlp_b3[o];for(int i=0;i<32;++i)z+=mlp_w3[o*32+i]*h2[i];corr[o]=tanhf(z)*scale[o];}
        State n=classic_s;n.v+=corr[0]*p.dt;n.vy+=corr[1]*p.dt;n.omega+=corr[2]*p.dt;
        n.x=s.x+(n.v*fast_cos(s.yaw)-n.vy*fast_sin(s.yaw))*p.dt;
        n.y=s.y+(n.v*fast_sin(s.yaw)+n.vy*fast_cos(s.yaw))*p.dt;n.yaw=angle_normalize(s.yaw+n.omega*p.dt);
        n.slip_angle=atan2f(n.vy,fabsf(n.v)+1e-5f);n.ay=(n.vy-s.vy)/p.dt+n.v*n.omega;
        for(int i=0;i<8;++i)command_history[i]=command_history[i+2];command_history[8]=u.steer;command_history[9]=speed_cmd;
        return n;
    }

    __device__ State update_kinematic_mlp_no_imu_residual(const State &s,const Control &u,
                                                           const Params &p,float *command_history)
    {
        State classic_s=update_kinematic(s,u,p);
        float speed_cmd=fminf(p.max_speed,fmaxf(p.min_speed,s.v+u.accel*p.dt));
        float raw[18]={s.v,s.vy,s.omega,u.steer,speed_cmd,
                       classic_s.v,classic_s.vy,classic_s.omega};
        for(int i=0;i<10;++i) raw[8+i]=command_history[i];
        const int ni[18]={0,1,2,3,4,5,6,7,3,4,3,4,3,4,3,4,3,4};
        float in[18],h1[64],h2[32];
        for(int i=0;i<18;++i) in[i]=(raw[i]-mlp_noimu_feature_mean[ni[i]])/mlp_noimu_feature_std[ni[i]];
        for(int o=0;o<64;++o){float z=mlp_noimu_b1[o];for(int i=0;i<18;++i)z+=mlp_noimu_w1[o*18+i]*in[i];h1[o]=z*sigmoidf_fast(z);}
        for(int o=0;o<32;++o){float z=mlp_noimu_b2[o];for(int i=0;i<64;++i)z+=mlp_noimu_w2[o*64+i]*h1[i];h2[o]=z*sigmoidf_fast(z);}
        float corr[3],scale[3]={8.f,8.f,30.f};
        for(int o=0;o<3;++o){float z=mlp_noimu_b3[o];for(int i=0;i<32;++i)z+=mlp_noimu_w3[o*32+i]*h2[i];corr[o]=tanhf(z)*scale[o];}
        State n=classic_s;n.v+=corr[0]*p.dt;n.vy+=corr[1]*p.dt;n.omega+=corr[2]*p.dt;
        n.x=s.x+(n.v*fast_cos(s.yaw)-n.vy*fast_sin(s.yaw))*p.dt;
        n.y=s.y+(n.v*fast_sin(s.yaw)+n.vy*fast_cos(s.yaw))*p.dt;n.yaw=angle_normalize(s.yaw+n.omega*p.dt);
        n.slip_angle=atan2f(n.vy,fabsf(n.v)+1e-5f);n.ay=(n.vy-s.vy)/p.dt+n.v*n.omega;
        for(int i=0;i<8;++i)command_history[i]=command_history[i+2];command_history[8]=u.steer;command_history[9]=speed_cmd;
        return n;
    }

    template<int N>
    __device__ void split_mlp(const float *raw,const float *mean,const float *std,
                              const float *w1,const float *b1,const float *w2,const float *b2,
                              const float *w3,const float *b3,float *out)
    {
        float in[N],h1[64],h2[32];
        for(int i=0;i<N;++i)in[i]=(raw[i]-mean[i])/std[i];
        for(int o=0;o<64;++o){float z=b1[o];for(int i=0;i<N;++i)z+=w1[o*N+i]*in[i];h1[o]=z*sigmoidf_fast(z);}
        for(int o=0;o<32;++o){float z=b2[o];for(int i=0;i<64;++i)z+=w2[o*64+i]*h1[i];h2[o]=z*sigmoidf_fast(z);}
        const float scale[3]={8.f,8.f,30.f};
        for(int o=0;o<3;++o){float z=b3[o];for(int i=0;i<32;++i)z+=w3[o*32+i]*h2[i];out[o]=tanhf(z)*scale[o];}
    }

    template<int N>
    __device__ void split_residual_mlp_v2(
        const float*,const float*,const float*,const float*,const float*,
        const float*,const float*,const float*,const float*,float*);

    __host__ __device__ inline bool uses_40ms_rollout_knot(int dynamics_model) {
        return dynamics_model == DYNAMIC_MLP_RESIDUAL_SERVO_LAG ||
               dynamics_model == DYNAMIC_SERVO_LAG ||
               dynamics_model == DYNAMIC_MLP_RESIDUAL_SERVO_LAG_VX_DELTA_24D;
    }

    // Real-car v2 residual contract: 20 -> 64 -> 32 -> 3, ReLU hidden
    // activations and an unbounded linear residual-derivative output. Keep the
    // legacy split_mlp above for checkpoints trained with SiLU+tanh scaling.
    template<int N>
    __device__ void split_residual_mlp_v2(
        const float *raw, const float *mean, const float *std,
        const float *w1, const float *b1, const float *w2, const float *b2,
        const float *w3, const float *b3, float *out)
    {
        float normalized[N], hidden1[64], hidden2[32];
        for (int i = 0; i < N; ++i)
            normalized[i] = (raw[i] - mean[i]) / std[i];
        for (int o = 0; o < 64; ++o) {
            float value = b1[o];
            for (int i = 0; i < N; ++i) value += w1[o*N+i] * normalized[i];
            hidden1[o] = fmaxf(value, 0.0f);
        }
        for (int o = 0; o < 32; ++o) {
            float value = b2[o];
            for (int i = 0; i < 64; ++i) value += w2[o*64+i] * hidden1[i];
            hidden2[o] = fmaxf(value, 0.0f);
        }
        for (int o = 0; o < 3; ++o) {
            float value = b3[o];
            for (int i = 0; i < 32; ++i) value += w3[o*32+i] * hidden2[i];
            out[o] = value;
        }
    }

    __device__ State update_kinematic_noslip_noimu_direct(const State &s,const Control &u,
                                                    const Params &p,float *history)
    {
        const float speed_cmd=fminf(p.max_speed,fmaxf(p.min_speed,u.accel));
        const float steer=fminf(.55f,fmaxf(-.55f,p.kinematic_steer_scale*u.steer+p.kinematic_steer_bias));
        const float base_ax=fminf(p.max_accel,fmaxf(p.min_accel,p.speed_servo_kp*(speed_cmd-s.v)));
        // min/max_speed constrain the command only. The predicted state is the
        // unconstrained result of integrating the bounded acceleration.
        const float base_v=s.v+base_ax*p.dt;
        const float target_w=s.v*tanf(steer)/(p.l_f+p.l_r);
        const float base_w=s.omega+fminf(p.kinematic_max_yaw_accel,fmaxf(-p.kinematic_max_yaw_accel,
            (target_w-s.omega)/fmaxf(p.kinematic_yaw_rate_time_constant,1e-3f)))*p.dt;
        float raw[16]={s.v,s.omega,u.steer,speed_cmd,base_v,base_w};
        for(int i=0;i<10;++i)raw[6+i]=history[i];
        float corr[3];split_mlp<16>(raw,knd_mean,knd_std,knd_w1,knd_b1,knd_w2,knd_b2,knd_w3,knd_b3,corr);
        State n=s;n.v=base_v+corr[0]*p.dt;n.vy=0.f;n.omega=base_w+corr[2]*p.dt;
        n.ax=(n.v-s.v)/p.dt;n.ay=(n.vy-s.vy)/p.dt+n.v*n.omega;
        n.x=s.x+(n.v*fast_cos(s.yaw)-n.vy*fast_sin(s.yaw))*p.dt;
        n.y=s.y+(n.v*fast_sin(s.yaw)+n.vy*fast_cos(s.yaw))*p.dt;n.yaw=angle_normalize(s.yaw+n.omega*p.dt);
        n.slip_angle=atan2f(n.vy,fabsf(n.v)+1e-5f);
        for(int i=0;i<8;++i)history[i]=history[i+2];history[8]=u.steer;history[9]=speed_cmd;return n;
    }

    // Slip-aware kinematic baseline. The MLP itself has no IMU feature; s.vy
    // is initialized once per control cycle by the persistent 2-state KF.
    __device__ State update_slip_kinematic_with_imu_direct(const State &s,const Control &u,
                                                            const Params &p,float *history)
    {
        const float speed_cmd=fminf(p.max_speed,fmaxf(p.min_speed,u.accel));
        const float steer_target=fminf(.55f,fmaxf(-.55f,p.kinematic_steer_scale*u.steer+p.kinematic_steer_bias));
        const float previous_command=history[8];
        const float previous_delta=history[10];
        const float steer_rate=fminf(p.actuator_max_steer_rate,fmaxf(-p.actuator_max_steer_rate,
            (steer_target-previous_delta)/fmaxf(p.steer_servo_time_constant,1e-3f)));
        const float steer=fminf(.55f,fmaxf(-.55f,previous_delta+steer_rate*p.dt));
        const float beta=atan2f(s.vy,s.v);
        const float base_ax=fminf(p.max_accel,fmaxf(p.min_accel,p.speed_servo_kp*(speed_cmd-s.v)));
        const float base_vx=s.v+base_ax*p.dt,base_vy=s.vy;
        const float target_w=base_vx*tanf(steer)/(p.l_f+p.l_r);
        const float base_w=s.omega+fminf(p.kinematic_max_yaw_accel,fmaxf(-p.kinematic_max_yaw_accel,
            (target_w-s.omega)/fmaxf(p.kinematic_yaw_rate_time_constant,1e-3f)))*p.dt;
        float raw[20]={s.v,s.vy,s.omega,u.steer,speed_cmd,steer,u.steer-previous_command,
                       base_vx,base_vy,base_w};
        for(int i=0;i<10;++i)raw[10+i]=history[i];
        float corr[3];split_mlp<20>(raw,ksd_mean,ksd_std,ksd_w1,ksd_b1,ksd_w2,ksd_b2,ksd_w3,ksd_b3,corr);
        State n=s;n.v=base_vx+corr[0]*p.dt;n.vy=base_vy+corr[1]*p.dt;n.omega=base_w+corr[2]*p.dt;
        n.ax=(n.v-s.v)/p.dt-s.vy*s.omega;n.ay=(n.vy-s.vy)/p.dt+s.v*s.omega;
        const float next_speed=hypotf(n.v,n.vy),next_beta=atan2f(n.vy,n.v);
        n.x=s.x+next_speed*cosf(s.yaw+next_beta)*p.dt;
        n.y=s.y+next_speed*sinf(s.yaw+next_beta)*p.dt;
        n.yaw=angle_normalize(s.yaw+n.omega*p.dt);n.slip_angle=next_beta;
        for(int i=0;i<8;++i)history[i]=history[i+2];history[8]=u.steer;history[9]=speed_cmd;history[10]=steer;return n;
    }

    __device__ State update_dynamic_imu_recursive(const State &s,const Control &u,
                                                   const Params &p,float *history)
    {
        const float speed_cmd=fminf(p.max_speed,fmaxf(p.min_speed,u.accel));
        const float steer=fminf(.55f,fmaxf(-.55f,p.kinematic_steer_scale*u.steer+p.kinematic_steer_bias));
        const float base_ax=fminf(p.max_accel,fmaxf(p.min_accel,p.speed_servo_kp*(speed_cmd-s.v)));
        const float base_w=s.v*tanf(steer)/(p.l_f+p.l_r),base_ay=s.v*base_w;
        float raw[20]={s.v,s.vy,s.omega,s.ax,s.ay,u.steer,speed_cmd,base_ax,base_ay,base_w};
        for(int i=0;i<10;++i)raw[10+i]=history[i];
        float pred[3];split_mlp<20>(raw,dir_mean,dir_std,dir_w1,dir_b1,dir_w2,dir_b2,dir_w3,dir_b3,pred);
        State n=s;n.ax=pred[0];n.ay=pred[1];n.omega=pred[2];
        n.v=s.v+(n.ax+s.vy*s.omega)*p.dt;n.vy=s.vy+(n.ay-s.v*s.omega)*p.dt;
        n.x=s.x+(n.v*fast_cos(s.yaw)-n.vy*fast_sin(s.yaw))*p.dt;
        n.y=s.y+(n.v*fast_sin(s.yaw)+n.vy*fast_cos(s.yaw))*p.dt;n.yaw=angle_normalize(s.yaw+n.omega*p.dt);
        n.slip_angle=atan2f(n.vy,fabsf(n.v)+1e-5f);
        for(int i=0;i<8;++i)history[i]=history[i+2];history[8]=u.steer;history[9]=speed_cmd;return n;
    }

    // Pacejka dynamic classic base + residual MLP. This is byte-for-byte the
    // Same 20-D feature order used by step_4_build_40ms_dataset.py and
    // step_5_train_residual_mlp.py.
    template <bool APPLY_RESIDUAL>
    __device__ State update_dynamic_mlp_residual(
        const State &current_state,
        const Control &control,
        const Params &params,
        float *command_history,
        bool use_servo_lag,
        float *vx_history = nullptr)
    {
        constexpr float MAX_STEERING_ANGLE = 0.55f;
        constexpr float MIN_DYNAMIC_SPEED = 0.5f;
        // The servo-lag checkpoint is a single 40 ms transition model.  ROS
        // still solves/publishes at 50 Hz; this dt only spaces rollout knots.
        const float dynamics_dt = use_servo_lag ? params.model_dt : params.dt;

        const float speed_command = fminf(
            params.max_speed,
            fmaxf(params.min_speed, control.accel));
        const float previous_steering_command = command_history[8];
        const float previous_actuator_steering_angle = command_history[10];

        // Training histories include the command represented by the current
        // feature row. Make the future rollout use that same causal contract.
        if (use_servo_lag) {
            for (int history_index = 0; history_index < 8; ++history_index)
                command_history[history_index] = command_history[history_index + 2];
            command_history[8] = control.steer;
            command_history[9] = speed_command;
        }

        float steering_angle;
        if (use_servo_lag) {
            const float target_steering_angle = fminf(
                MAX_STEERING_ANGLE,
                fmaxf(-MAX_STEERING_ANGLE,
                      params.kinematic_steer_scale * control.steer
                          + params.kinematic_steer_bias));
            const float steering_rate = fminf(
                params.actuator_max_steer_rate,
                fmaxf(-params.actuator_max_steer_rate,
                      (target_steering_angle - previous_actuator_steering_angle)
                          / fmaxf(params.steer_servo_time_constant, 1.0e-3f)));
            steering_angle = fminf(
                MAX_STEERING_ANGLE,
                fmaxf(-MAX_STEERING_ANGLE,
                      previous_actuator_steering_angle + steering_rate * dynamics_dt));
        } else {
            // Causal no-lag contract: the command available at prediction
            // time t is the previous Ackermann steering command. Do not apply
            // servo scale/bias/tau in this model.
            steering_angle = fminf(
                MAX_STEERING_ANGLE,
                fmaxf(-MAX_STEERING_ANGLE, previous_steering_command));
        }

        float &actuator_speed_reference = command_history[11];
        if (use_servo_lag) {
            const float speed_reference_time_constant =
                speed_command >= actuator_speed_reference
                    ? params.speed_reference_accel_time_constant
                    : params.speed_reference_brake_time_constant;
            const float speed_reference_rate = fminf(
                params.actuator_max_speed_reference_rate,
                fmaxf(-params.actuator_max_speed_reference_rate,
                      (speed_command - actuator_speed_reference)
                        / fmaxf(speed_reference_time_constant, 1.0e-3f)));
            actuator_speed_reference += speed_reference_rate * dynamics_dt;
        } else {
            // Keep the auxiliary state coherent for diagnostics/model changes,
            // while the no-lag acceleration uses speed_command directly.
            actuator_speed_reference = speed_command;
        }
        const float longitudinal_speed_reference =
            use_servo_lag ? actuator_speed_reference : speed_command;
        const float classic_longitudinal_acceleration = fminf(
            params.max_accel,
            fmaxf(params.min_accel,
                  params.speed_servo_kp *
                    (longitudinal_speed_reference - current_state.v)));
        const float safe_longitudinal_velocity = fmaxf(
            fabsf(current_state.v), MIN_DYNAMIC_SPEED);

        const float front_tire_slip_angle = steering_angle - atan2f(
            current_state.vy + params.l_f * current_state.omega,
            safe_longitudinal_velocity);
        const float rear_tire_slip_angle = -atan2f(
            current_state.vy - params.l_r * current_state.omega,
            safe_longitudinal_velocity);

        // Pacejka inner term: B*alpha - E*(B*alpha - atan(B*alpha)).
        const float front_stiffness_times_slip =
            params.dynamic_mlp_B_f * front_tire_slip_angle;
        const float rear_stiffness_times_slip =
            params.dynamic_mlp_B_r * rear_tire_slip_angle;
        const float front_pacejka_inner = front_stiffness_times_slip
            - params.dynamic_mlp_E_f
                * (front_stiffness_times_slip - atanf(front_stiffness_times_slip));
        const float rear_pacejka_inner = rear_stiffness_times_slip
            - params.dynamic_mlp_E_r
                * (rear_stiffness_times_slip - atanf(rear_stiffness_times_slip));
        const float front_lateral_tire_force =
            params.F_zf * params.dynamic_mlp_D_f
            * sinf(params.dynamic_mlp_C_f * atanf(front_pacejka_inner));
        const float rear_lateral_tire_force =
            params.F_zr * params.dynamic_mlp_D_r
            * sinf(params.dynamic_mlp_C_r * atanf(rear_pacejka_inner));
        const float dynamic_lateral_acceleration =
            (front_lateral_tire_force * cosf(steering_angle)
             + rear_lateral_tire_force) / params.mass;
        const float classic_yaw_moment =
            params.l_f * front_lateral_tire_force * cosf(steering_angle)
            - params.l_r * rear_lateral_tire_force;
        const float dynamic_yaw_acceleration =
            classic_yaw_moment / params.dynamic_mlp_I_z;
        const float blend_input = fminf(1.0f, fmaxf(0.0f,
            (fabsf(current_state.v) - 0.2f) / 0.3f));
        const float dynamic_blend = blend_input * blend_input
            * (3.0f - 2.0f * blend_input);
        constexpr float LOW_SPEED_TIME_CONSTANT = 0.1f;
        const float kinematic_yaw_rate = current_state.v * tanf(steering_angle)
            / fmaxf(params.l_f + params.l_r, 1.0e-6f);
        const float kinematic_lateral_acceleration =
            current_state.v * current_state.omega
            - current_state.vy / LOW_SPEED_TIME_CONSTANT;
        const float kinematic_yaw_acceleration =
            (kinematic_yaw_rate - current_state.omega)
            / LOW_SPEED_TIME_CONSTANT;
        const float classic_lateral_acceleration =
            dynamic_blend * dynamic_lateral_acceleration
            + (1.0f - dynamic_blend) * kinematic_lateral_acceleration;
        const float classic_yaw_acceleration =
            dynamic_blend * dynamic_yaw_acceleration
            + (1.0f - dynamic_blend) * kinematic_yaw_acceleration;

        // Body-frame equations: ax=d(vx)/dt-vy*r, ay=d(vy)/dt+vx*r.
        const float classic_next_longitudinal_velocity = current_state.v
            + (classic_longitudinal_acceleration
               + current_state.vy * current_state.omega) * dynamics_dt;
        const float classic_next_lateral_velocity = current_state.vy
            + (classic_lateral_acceleration
               - current_state.v * current_state.omega) * dynamics_dt;
        const float classic_next_yaw_rate = current_state.omega
            + classic_yaw_acceleration * dynamics_dt;

        // DYNAMIC_SERVO_LAG is the non-MLP deployment model.  Do not build
        // the 22-D residual feature vector or execute the residual-output
        // path when the caller explicitly disabled the learned correction.
        // This branch preserves the same actuator/history contract as the
        // residual path, but only integrates the Pacejka/kinematic baseline.
        if constexpr (!APPLY_RESIDUAL) {
            State next_state = current_state;
            next_state.v = classic_next_longitudinal_velocity;
            next_state.vy = classic_next_lateral_velocity;
            next_state.omega = classic_next_yaw_rate;
            next_state.ax = classic_longitudinal_acceleration;
            next_state.ay = classic_lateral_acceleration;

            const float next_speed = hypotf(next_state.v, next_state.vy);
            const float next_body_slip_angle = atan2f(next_state.vy, next_state.v);
            next_state.x = current_state.x + next_speed
                * cosf(current_state.yaw + next_body_slip_angle) * dynamics_dt;
            next_state.y = current_state.y + next_speed
                * sinf(current_state.yaw + next_body_slip_angle) * dynamics_dt;
            next_state.yaw = angle_normalize(
                current_state.yaw + next_state.omega * dynamics_dt);
            next_state.slip_angle = next_body_slip_angle;

            if (!use_servo_lag) {
                for (int history_index = 0; history_index < 8; ++history_index)
                    command_history[history_index] = command_history[history_index + 2];
                command_history[8] = control.steer;
                command_history[9] = speed_command;
            }
            command_history[10] = steering_angle;
            return next_state;
        }

        // Keep this exact 22-D order synchronized with the canonical 40 ms
        // residual dataset/training pipeline:
        // [vx, vy, yaw_rate, steer_cmd, speed_cmd, applied_steer,
        //  steer_cmd_delta, classic_next_vx, classic_next_vy,
        //  classic_next_yaw_rate, five previous (steer_cmd, speed_cmd) pairs,
        //  current_ax, current_ay]. At the first knot ax/ay are the causally
        //  aligned IMU measurements; later knots use the predicted values.
        float mlp_features[22] = {
            current_state.v,
            current_state.vy,
            current_state.omega,
            control.steer,
            speed_command,
            steering_angle,
            control.steer - previous_steering_command,
            classic_next_longitudinal_velocity,
            classic_next_lateral_velocity,
            classic_next_yaw_rate
        };
        for (int feature_index = 0; feature_index < 10; ++feature_index) {
            mlp_features[10 + feature_index] = command_history[feature_index];
        }
        mlp_features[20]=current_state.ax;
        mlp_features[21]=current_state.ay;

        // MLP outputs residual derivatives [delta_ax, delta_ay, delta_yaw_accel].
        float residual_derivatives[3];
        if (vx_history != nullptr) {
            float features_24d[24];
            for (int i=0;i<20;++i) features_24d[i]=mlp_features[i];
            for (int i=0;i<4;++i) features_24d[20+i]=vx_history[i+1]-vx_history[i];
            split_residual_mlp_v2<24>(features_24d,dmr24_mean,dmr24_std,
                dmr24_w1,dmr24_b1,dmr24_w2,dmr24_b2,dmr24_w3,dmr24_b3,
                residual_derivatives);
        } else {
            split_residual_mlp_v2<22>(mlp_features,dmr_mean,dmr_std,
                dmr_w1,dmr_b1,dmr_w2,dmr_b2,dmr_w3,dmr_b3,
                residual_derivatives);
        }

        // Bound only the learned correction to its trained output domain.
        // Sensor values and the resulting total state are never clamped here.
        residual_derivatives[0] = fminf(params.mlp_max_residual_ax,
            fmaxf(-params.mlp_max_residual_ax, residual_derivatives[0]));
        residual_derivatives[1] = fminf(params.mlp_max_residual_ay,
            fmaxf(-params.mlp_max_residual_ay, residual_derivatives[1]));
        residual_derivatives[2] = fminf(params.mlp_max_residual_yaw_accel,
            fmaxf(-params.mlp_max_residual_yaw_accel, residual_derivatives[2]));
        // The residual network is trained on dynamic-driving data. Fade its
        // correction out with the same low-speed transition as the base model.
        residual_derivatives[0] *= dynamic_blend;
        residual_derivatives[1] *= dynamic_blend;
        residual_derivatives[2] *= dynamic_blend;

        State next_state = current_state;
        next_state.v = classic_next_longitudinal_velocity
            + residual_derivatives[0] * dynamics_dt;
        next_state.vy = classic_next_lateral_velocity
            + residual_derivatives[1] * dynamics_dt;
        next_state.omega = classic_next_yaw_rate
            + residual_derivatives[2] * dynamics_dt;
        next_state.ax = classic_longitudinal_acceleration + residual_derivatives[0];
        next_state.ay = classic_lateral_acceleration + residual_derivatives[1];

        const float next_speed = hypotf(next_state.v, next_state.vy);
        const float next_body_slip_angle = atan2f(next_state.vy, next_state.v);
        next_state.x = current_state.x + next_speed
            * cosf(current_state.yaw + next_body_slip_angle) * dynamics_dt;
        next_state.y = current_state.y + next_speed
            * sinf(current_state.yaw + next_body_slip_angle) * dynamics_dt;
        next_state.yaw = angle_normalize(
            current_state.yaw + next_state.omega * dynamics_dt);
        next_state.slip_angle = next_body_slip_angle;

        if (!use_servo_lag) {
            for (int history_index = 0; history_index < 8; ++history_index)
                command_history[history_index] = command_history[history_index + 2];
            command_history[8] = control.steer;
            command_history[9] = speed_command;
        }
        command_history[10] = steering_angle;
        if (vx_history != nullptr) {
            for (int i=0;i<4;++i) vx_history[i]=vx_history[i+1];
            vx_history[4]=next_state.v;
        }
        return next_state;
    }

    __global__ void debug_dynamic_mlp_residual_step_kernel(
        State state, Control control, Params params,
        const float *history_input, State *state_output, float *history_output,
        bool use_servo_lag)
    {
        if (blockIdx.x != 0 || threadIdx.x != 0) return;
        float history[12];
        for (int i = 0; i < 12; ++i) history[i] = history_input[i];
        *state_output = update_dynamic_mlp_residual<true>(
            state, control, params, history, use_servo_lag);
        for (int i = 0; i < 12; ++i) history_output[i] = history[i];
    }

    __global__ void debug_dynamic_mlp_vx_delta_step_kernel(
        State state,Control control,Params params,const float *history_input,
        const float *vx_input,State *state_output,float *history_output,float *vx_output)
    {
        if(blockIdx.x||threadIdx.x)return;
        float history[12],vx_history[5];
        for(int i=0;i<12;++i)history[i]=history_input[i];
        for(int i=0;i<5;++i)vx_history[i]=vx_input[i];
        *state_output=update_dynamic_mlp_residual<true>(
            state,control,params,history,true,vx_history);
        for(int i=0;i<12;++i)history_output[i]=history[i];
        for(int i=0;i<5;++i)vx_output[i]=vx_history[i];
    }

    __host__ __device__ State update_legacy_hybrid(const State &s, const Control &u, const Params &p)
    {
        float px = s.x; float py = s.y; float yaw = s.yaw;
        float vel = s.v; float omega = s.omega; float slip_angle = s.slip_angle;
        
        // 1. 저속 구간: 순수 운동학 모델 (Kinematic Model)
        // 분모에 vel이 들어가는 동역학의 특이점(Division by zero)을 방지
        if (fabsf(vel) < 0.5f) {
            State next_s;
            float wheelbase = p.l_f + p.l_r;

            // 기존 MPC Kinematic 수식 적용
            float dot_x = vel * fast_cos(yaw);
            float dot_y = vel * fast_sin(yaw);
            float dot_yaw = vel * tanf(u.steer) / wheelbase;
            float dot_vel = u.accel;

            next_s.x = px + dot_x * p.dt;
            next_s.y = py + dot_y * p.dt;
            next_s.yaw = angle_normalize(yaw + dot_yaw * p.dt);
            next_s.v = vel + dot_vel * p.dt;

            // 기존 MPC 기준 미사용 변수 초기화 (omega는 제어 연속성을 위해 dot_yaw 인가)
            next_s.omega = dot_yaw; 
            next_s.slip_angle = 0.0f;
            
            // MPPI 비용 함수용 보조 변수
            next_s.vy = 0.0f; 
            next_s.ax = dot_vel;
            next_s.ay = 0.0f;

            return next_s;
        }

        // 2. 고속 구간: 파세이카 동역학 모델 (Pacejka Dynamic Model)
        
        // 슬립각(beta)으로부터 차량 좌표계 vx, vy 역산 (타이어 슬립각 alpha 연산용)
        float vx = vel * fast_cos(slip_angle);
        float vy = vel * fast_sin(slip_angle);

        // 전/후륜 타이어 슬립각 계산
        float alpha_f = u.steer - atan2f(vy + p.l_f * omega, vx);
        float alpha_r = -atan2f(vy - p.l_r * omega, vx);

        // 타이어 횡력 연산 — ForzaETH On-Track-SysID 와 동일한 4-파라미터 매직 포뮬러.
        //   F_y = F_z * D * sin( C * atan( B*a - E*(B*a - atan(B*a)) ) )
        // D 는 무차원(마찰계수), 힘의 크기는 정하중 F_z 가 만든다.
        // E=0 이면 안쪽 괄호가 B*a 로 환원되어 이전 3-파라미터 수식과 완전히 동일해진다.
        float bf_a = p.B_f * alpha_f;
        float br_a = p.B_r * alpha_r;
        float F_fy = p.F_zf * p.D_f * fast_sin(p.C_f * atanf(bf_a - p.E_f * (bf_a - atanf(bf_a))));
        float F_ry = p.F_zr * p.D_r * fast_sin(p.C_r * atanf(br_a - p.E_r * (br_a - atanf(br_a))));

        // 기존 MPC 동역학 수식 완벽 적용 (단위 및 로직 동일)
        float dot_x = vel * fast_cos(yaw + slip_angle);
        float dot_y = vel * fast_sin(yaw + slip_angle);
        float dot_yaw = omega;
        float dot_vel = u.accel * (1.0f - p.Cm0 * vel); // 모터 감쇠 계수 적용
        float dot_omega = (p.l_f * F_fy * fast_cos(u.steer) - p.l_r * F_ry) / p.I_z; // 토크 / 관성모멘트
        float dot_slip = ((F_fy + F_ry) / (p.mass * vel)) - omega; // 횡력의 합 / (질량 * 속도)

        // 상태 업데이트 (Euler Integration)
        State next_s;
        next_s.x = px + dot_x * p.dt;
        next_s.y = py + dot_y * p.dt;
        next_s.yaw = angle_normalize(yaw + dot_yaw * p.dt);
        next_s.v = vel + dot_vel * p.dt;
        next_s.ax = dot_vel;
        next_s.omega = omega + dot_omega * p.dt;
        next_s.slip_angle = slip_angle + dot_slip * p.dt;

        // MPPI 비용 함수에서 사용하는 보조 변수 도출
        next_s.vy = next_s.v * fast_sin(next_s.slip_angle); // 슬립각 기반 vy 
        next_s.ay = (F_fy * fast_cos(u.steer) + F_ry) / p.mass; // 횡가속도 a_y = F_y / m 

        return next_s;
    }

    __host__ __device__ State update_dynamics(const State &s, const Control &u, const Params &p)
    {
        if (p.dynamics_model == KINEMATIC) {
            return update_kinematic(s, u, p);
        }

        // The trained GRU residual is intentionally not approximated here. Its
        // 96-dimensional recurrent state must be owned by each CUDA rollout;
        // silently calling the kinematic model would make YAML claim a model
        // that is not actually running.
        if (p.dynamics_model == KINEMATIC_RESIDUAL) {
            return update_kinematic(s, u, p);
        }
        return update_legacy_hybrid(s, u, p);
    }

    __device__ float compute_cost_cuda(
        const State &s,
        const float *ref_xs, const float *ref_ys, const float *ref_yaws, int path_len,
        const Control &u, const Control &u_prev,
        const Params &p,
        float min_bnd_dist,
        float applied_steering_angle,
        bool terminal_step,
        int horizon_step,
        const float* dynamic_obs_x,
        const float* dynamic_obs_y,
        const float* dynamic_obs_yaw,
        const float* dynamic_obs_semi_major,
        const float* dynamic_obs_semi_minor,
        const std::uint8_t* dynamic_obs_is_dynamic,
        int dynamic_obstacle_count,
        int dynamic_obstacle_horizon,
        int* last_idx)
    {
        float min_dist_sq = 1e9f;
        int nearest_idx = -1;

        int start_search = *last_idx; 
        int search_window = 30; 
        
        if (start_search >= path_len) start_search %= path_len;
        if (start_search < 0) start_search = 0;

        for (int offset = 0; offset < search_window; ++offset)
        {
            int i = start_search + offset;
            if (i >= path_len) i -= path_len; 

            float dx = s.x - ref_xs[i];
            float dy = s.y - ref_ys[i];
            float dist_sq = dx * dx + dy * dy;

            if (dist_sq < min_dist_sq)
            {
                min_dist_sq = dist_sq;
                nearest_idx = i;
            }
        }

        if (nearest_idx == -1) nearest_idx = start_search;
        *last_idx = nearest_idx; 

        // 1. MPCC path errors
        // 논문의 contouring/lag error를 경로 접선 좌표계에 투영한다.
        // contour는 racing line 탐색을 허용하도록 작게, lag는 기하 progress가
        // 실제 차량 위치보다 앞서 나가는 것을 막도록 상대적으로 크게 둔다.
        float dx_ref = s.x - ref_xs[nearest_idx];
        float dy_ref = s.y - ref_ys[nearest_idx];
        float ref_cos = fast_cos(ref_yaws[nearest_idx]);
        float ref_sin = fast_sin(ref_yaws[nearest_idx]);
        float contour_error = -ref_sin * dx_ref + ref_cos * dy_ref;
        float lag_error = ref_cos * dx_ref + ref_sin * dy_ref;
        float heading_error = angle_normalize(s.yaw - ref_yaws[nearest_idx]);
        float dist_error = min_dist_sq;  // 이전 설정과의 호환용
        float path_cost = p.q_dist * dist_error
                        + p.q_contour * contour_error * contour_error
                        + p.q_lag * lag_error * lag_error
                        + p.q_heading * heading_error * heading_error;

        // 2. 랩타임 속도 비용
        // 경로 접선 방향 속도는 보상하되 max_speed를 넘는 속도는 제곱 비용으로
        // 억제한다.
        float forward_v = s.v * fast_cos(s.yaw - ref_yaws[nearest_idx]);
        float overspeed = fmaxf(0.0f, s.v - p.max_speed);
        float vel_cost = -(p.q_v * 0.2f) * forward_v
                       +  p.q_v * overspeed * overspeed;
        // Do not reward high speed while the vehicle is still recovering its
        // lateral position or heading. This couples recovery urgency to speed
        // without introducing a discontinuous speed cap.
        float error_speed_cost = p.q_error_speed * s.v * s.v
            * (contour_error * contour_error + heading_error * heading_error);

        // Combined longitudinal/lateral tire utilization.  Penalizing ay
        // alone incorrectly treats simultaneous braking and cornering as
        // independent.  The normalized ellipse is dimensionless and starts
        // charging as soon as its boundary is exceeded.
        const float ax_limit = fmaxf(p.longitudinal_accel_soft_limit, 1.0e-3f);
        const float ay_limit = fmaxf(p.lat_g_soft_limit, 1.0e-3f);
        const float normalized_ax = s.ax / ax_limit;
        const float normalized_ay = s.ay / ay_limit;
        const float tire_utilization = normalized_ax * normalized_ax
                                     + normalized_ay * normalized_ay;
        const float friction_excess = fmaxf(0.0f, tire_utilization - 1.0f);
        const float friction_ellipse_cost = p.q_lat_g * friction_excess * friction_excess;

        // Front-tire slip soft constraint. Use the same servo-lagged steering
        // angle and sign convention as update_dynamic_mlp_residual().
        constexpr float MIN_FRONT_SLIP_SPEED = 0.5f;
        const float safe_front_vx = fmaxf(fabsf(s.v), MIN_FRONT_SLIP_SPEED);
        const float front_tire_slip_angle = applied_steering_angle - atan2f(
            s.vy + p.l_f * s.omega, safe_front_vx);
        const float front_slip_excess = fmaxf(
            0.0f, fabsf(front_tire_slip_angle) - p.front_slip_soft_limit);
        const float front_speed_ratio = fminf(
            1.0f, fabsf(s.v) / fmaxf(p.front_slip_cost_min_speed, 1.0e-3f));
        const float front_speed_gate = front_speed_ratio * front_speed_ratio
            * (3.0f - 2.0f * front_speed_ratio);
        const float front_slip_cost =
            front_speed_gate * p.q_front_slip
            * front_slip_excess * front_slip_excess;

        // Rear-tire slip soft constraint. Unlike the GG ellipse, this catches
        // rear-axle saturation/oversteer even when total vehicle acceleration
        // remains inside the combined-grip boundary. Disable it at low speed,
        // where dividing by vx makes the inferred slip angle ill-conditioned.
        constexpr float MIN_REAR_SLIP_SPEED = 0.5f;
        const float safe_vx = fmaxf(fabsf(s.v), MIN_REAR_SLIP_SPEED);
        const float rear_tire_slip_angle = -atan2f(
            s.vy - p.l_r * s.omega, safe_vx);
        const float rear_slip_excess = fmaxf(
            0.0f, fabsf(rear_tire_slip_angle) - p.rear_slip_soft_limit);
        const float rear_speed_ratio = fminf(
            1.0f, fabsf(s.v) / fmaxf(p.rear_slip_cost_min_speed, 1.0e-3f));
        const float rear_speed_gate = rear_speed_ratio * rear_speed_ratio
            * (3.0f - 2.0f * rear_speed_ratio);
        const float rear_slip_cost =
            rear_speed_gate * p.q_rear_slip
            * rear_slip_excess * rear_slip_excess;

        // 4. Control Input Cost
        float d_steer = u.steer - u_prev.steer;
        float d_accel = u.accel - u_prev.accel;
        float rate_cost = p.q_du * (d_steer * d_steer + d_accel * d_accel);
        float steer_cost = p.q_steer * (u.steer * u.steer);
        
        // 6. Track-boundary soft constraint.
        //
        // The reference MPC introduces s_k >= 0 in both lane inequalities
        // and minimizes q_s*s_k^2.  For a sampled MPPI state the optimal
        // slack is available analytically, so no solver variable or extra
        // kernel is required:
        //   s_k = max(0, required_clearance - measured_clearance).
        // A negative measured clearance simply means the predicted state is
        // outside the lane and produces a correspondingly larger finite cost.
        // It is deliberately not converted into an invalid rollout.
        const float boundary_slack =
            fmaxf(0.0f, p.collision_radius - min_bnd_dist);
        const float boundary_slack_weight = terminal_step
            ? p.q_boundary_terminal_slack : p.q_boundary_slack;
        const float boundary_cost = boundary_slack_weight
            * boundary_slack * boundary_slack;

        // 7. Obstacle soft constraint. car_radius remains the hard physical
        // guard, but the smooth cost begins outside it. Without this margin the
        // cost is exactly zero until the hard post-selection suddenly replaces
        // the entire weighted sequence.
        float obs_cost = 0.0f;
        const float obstacle_soft_radius =
            p.car_radius + p.obstacle_soft_margin;
        for (int i = 0; i < p.num_obstacles; ++i) {
            float dx = s.x - p.obs_x[i];
            float dy = s.y - p.obs_y[i];
            float dist = sqrtf(dx * dx + dy * dy);
            const float obstacle_slack=fmaxf(0.0f,obstacle_soft_radius-dist);
            obs_cost += p.q_obs*obstacle_slack*obstacle_slack;
        }
        if (dynamic_obstacle_count > 0 && dynamic_obstacle_horizon > 0) {
            const int step = min(horizon_step, dynamic_obstacle_horizon - 1);
            for (int i = 0; i < dynamic_obstacle_count; ++i) {
                const int index = i * dynamic_obstacle_horizon + step;
                const float dx = s.x - dynamic_obs_x[index];
                const float dy = s.y - dynamic_obs_y[index];
                const float distance = sqrtf(dx * dx + dy * dy);
                float boundary_radius = p.car_radius + p.obstacle_soft_margin;
                if (dynamic_obs_is_dynamic[i]) {
                    const float c = fast_cos(dynamic_obs_yaw[index]);
                    const float sn = fast_sin(dynamic_obs_yaw[index]);
                    const float local_x = c * dx + sn * dy;
                    const float local_y = -sn * dx + c * dy;
                    // The predictor axes already include the complete vehicle
                    // collision radius. Only the configurable soft margin is
                    // added here; adding car_radius again double-inflates it.
                    const float major = fmaxf(1.0e-3f,
                        dynamic_obs_semi_major[index] + p.obstacle_soft_margin);
                    const float minor = fmaxf(1.0e-3f,
                        dynamic_obs_semi_minor[index] + p.obstacle_soft_margin);
                    if (distance > 1.0e-6f) {
                        const float ux = local_x / distance;
                        const float uy = local_y / distance;
                        boundary_radius = rsqrtf(
                            ux * ux / (major * major) + uy * uy / (minor * minor));
                    } else {
                        boundary_radius = fmaxf(major, minor);
                    }
                }
                const float slack = fmaxf(0.0f, boundary_radius - distance);
                obs_cost += p.q_obs * slack * slack;
            }
        }

        const float tracking_cost=p.objective_mode==LMPC_OBJECTIVE
            ? 0.0f : path_cost+vel_cost+error_speed_cost;
        return tracking_cost + friction_ellipse_cost
             + front_slip_cost
             + rear_slip_cost
             + steer_cost + rate_cost + boundary_cost + obs_cost;
    }

    // 두 경로 인덱스 사이의 실제 중심선 arc length. 경로점 개수 대신 m 단위
    // 진행도를 쓰므로 centerline CSV의 샘플 간격이 바뀌어도 비용 의미가 유지된다.
    __device__ float compute_progress_distance(
        const State &start_state, const State &end_state,
        int start_idx, int end_idx,
        const float *ref_xs, const float *ref_ys, const float *ref_yaws,
        int path_len)
    {
        if (path_len <= 1) return 0.0f;

        int steps = end_idx - start_idx;
        if (steps < 0) steps += path_len;
        // 한 번의 1초 horizon에서 반 바퀴 이상 순간이동한 인덱스는 역주행/오탐이다.
        if (steps > path_len / 2) return 0.0f;

        float distance = 0.0f;
        int prev = start_idx;
        for (int n = 0; n < steps; ++n) {
            int next = prev + 1;
            if (next >= path_len) next = 0;
            float dx = ref_xs[next] - ref_xs[prev];
            float dy = ref_ys[next] - ref_ys[prev];
            distance += sqrtf(dx * dx + dy * dy);
            prev = next;
        }
        // 최근접 waypoint 사이의 sub-sample 진행도를 접선 투영으로 보간한다.
        // 이 항이 없으면 약 0.1 m마다 비용이 계단식으로 변한다.
        float start_dx = start_state.x - ref_xs[start_idx];
        float start_dy = start_state.y - ref_ys[start_idx];
        float end_dx = end_state.x - ref_xs[end_idx];
        float end_dy = end_state.y - ref_ys[end_idx];
        float start_lag = start_dx * fast_cos(ref_yaws[start_idx])
                        + start_dy * fast_sin(ref_yaws[start_idx]);
        float end_lag = end_dx * fast_cos(ref_yaws[end_idx])
                      + end_dy * fast_sin(ref_yaws[end_idx]);

        // 투영 오차가 비정상적으로 큰 경우 progress 보상을 악용하지 못하게 제한.
        start_lag = fminf(fmaxf(start_lag, -0.15f), 0.15f);
        end_lag = fminf(fmaxf(end_lag, -0.15f), 0.15f);
        return fmaxf(0.0f, distance + end_lag - start_lag);
    }

    // Frank-Wolfe solves the same small convex terminal problem used by LMPC:
    //   min_lambda q_ts ||x_N-S lambda||_T^2 + c Q^T lambda
    //   subject to lambda >= 0, sum(lambda)=1.
    // Only px/py/yaw participate, matching control/MPC/src/lmpc.cpp.
    __device__ float compute_lmpc_terminal_cost(const State &x, const Params &p)
    {
        const int n=p.safe_set_count;
        if(p.objective_mode!=LMPC_OBJECTIVE || n<=0) return 0.0f;
        int best=0;
        float best_value=1.0e30f;
        for(int j=0;j<n;++j) {
            const float dx=(x.x-p.safe_set_x[j])*p.safe_set_inv_x_scale;
            const float dy=(x.y-p.safe_set_y[j])*p.safe_set_inv_y_scale;
            const float da=angle_normalize(x.yaw-p.safe_set_yaw[j])*
                           p.safe_set_inv_yaw_scale;
            const float value=p.q_terminal_safe_set_slack*(dx*dx+dy*dy+da*da)
                +p.safe_set_cost_coefficient*p.safe_set_cost[j];
            if(value<best_value){best_value=value;best=j;}
        }
        float zx=p.safe_set_x[best],zy=p.safe_set_y[best];
        float za=x.yaw-angle_normalize(x.yaw-p.safe_set_yaw[best]);
        float zq=p.safe_set_cost[best];
        // Eight iterations are enough for this 3-D, 40-vertex local hull and
        // avoid introducing a per-rollout QP solver or dynamic allocation.
        for(int iteration=0;iteration<8;++iteration) {
            const float ex=(zx-x.x)*p.safe_set_inv_x_scale;
            const float ey=(zy-x.y)*p.safe_set_inv_y_scale;
            const float ea=angle_normalize(za-x.yaw)*p.safe_set_inv_yaw_scale;
            int vertex=0;float minimum=1.0e30f;
            for(int j=0;j<n;++j) {
                const float jyaw=x.yaw-angle_normalize(x.yaw-p.safe_set_yaw[j]);
                const float gradient=2.0f*p.q_terminal_safe_set_slack*(
                    ex*p.safe_set_inv_x_scale*p.safe_set_x[j]
                   +ey*p.safe_set_inv_y_scale*p.safe_set_y[j]
                   +ea*p.safe_set_inv_yaw_scale*jyaw)
                   +p.safe_set_cost_coefficient*p.safe_set_cost[j];
                if(gradient<minimum){minimum=gradient;vertex=j;}
            }
            const float vyaw=x.yaw-angle_normalize(x.yaw-p.safe_set_yaw[vertex]);
            const float dx=(p.safe_set_x[vertex]-zx)*p.safe_set_inv_x_scale;
            const float dy=(p.safe_set_y[vertex]-zy)*p.safe_set_inv_y_scale;
            const float da=(vyaw-za)*p.safe_set_inv_yaw_scale;
            const float dq=p.safe_set_cost[vertex]-zq;
            const float denominator=2.0f*p.q_terminal_safe_set_slack*(dx*dx+dy*dy+da*da);
            float gamma=denominator>1.0e-9f ? -(2.0f*p.q_terminal_safe_set_slack*
                (ex*dx+ey*dy+ea*da)+p.safe_set_cost_coefficient*dq)/denominator : 0.0f;
            gamma=fminf(1.0f,fmaxf(0.0f,gamma));
            zx+=gamma*(p.safe_set_x[vertex]-zx);zy+=gamma*(p.safe_set_y[vertex]-zy);
            za+=gamma*(vyaw-za);zq+=gamma*dq;
        }
        const float dx=(x.x-zx)*p.safe_set_inv_x_scale;
        const float dy=(x.y-zy)*p.safe_set_inv_y_scale;
        const float da=angle_normalize(x.yaw-za)*p.safe_set_inv_yaw_scale;
        return p.q_terminal_safe_set_slack*(dx*dx+dy*dy+da*da)
            +p.safe_set_cost_coefficient*zq;
    }
    
    // [수정된 함수] O(N) 바운더리 탐색을 대체하는 O(1) 횡방향 오차 기반 거리 연산
    __device__ float compute_min_boundary_distance(
        const State &s,
        const float *ref_xs, const float *ref_ys, const float *ref_yaws,
        const float *left_xs, const float *left_ys,
        const float *right_xs, const float *right_ys,
        int path_len, int current_path_idx, int *nearest_idx_out) 
    {
        if (ref_xs == nullptr || left_xs == nullptr || path_len <= 0) return 1e9f;

        // 1. 이전 인덱스 근처에서 가장 가까운 중심점(Reference) 탐색
        float min_dist_sq = 1e9f;
        int nearest_idx = current_path_idx;
        int search_window = 30;
        int start_search = current_path_idx;

        if (start_search >= path_len) start_search %= path_len;
        if (start_search < 0) start_search = 0;

        for (int offset = 0; offset < search_window; ++offset) {
            int i = start_search + offset;
            if (i >= path_len) i -= path_len;
            
            float dx = s.x - ref_xs[i];
            float dy = s.y - ref_ys[i];
            float dist_sq = dx * dx + dy * dy;
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                nearest_idx = i;
            }
        }
        
        // 찾은 인덱스를 외부로 반환하여 compute_cost_cuda의 탐색 속도도 높임
        if (nearest_idx_out != nullptr) *nearest_idx_out = nearest_idx;

        // 2. 중심선 기준 법선 벡터를 통한 횡방향 편차(e_y) 도출
        float dx = s.x - ref_xs[nearest_idx];
        float dy = s.y - ref_ys[nearest_idx];
        float ref_yaw = ref_yaws[nearest_idx];
        
        float nx = -fast_sin(ref_yaw);
        float ny = fast_cos(ref_yaw);
        float e_y = dx * nx + dy * ny;

        // 3. 해당 지점의 실제 좌우 도로 폭 연산
        float dx_l = left_xs[nearest_idx] - ref_xs[nearest_idx];
        float dy_l = left_ys[nearest_idx] - ref_ys[nearest_idx];
        float w_left = sqrtf(dx_l * dx_l + dy_l * dy_l);

        float dx_r = right_xs[nearest_idx] - ref_xs[nearest_idx];
        float dy_r = right_ys[nearest_idx] - ref_ys[nearest_idx];
        float w_right = sqrtf(dx_r * dx_r + dy_r * dy_r);

        // 4. 차량에서 양쪽 바운더리까지의 최단 거리 반환
        return fminf(w_left - e_y, w_right + e_y);
    }

    __global__ void init_rng_kernel(curandState *states, long seed, int K, int T)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < K * T) curand_init(seed, idx, 0, &states[idx]);
    }

    template <bool APPLY_RESIDUAL>
    __global__ void rollout_kernel(
        State *states, Control *controls, float *costs,
        float *min_boundary_clearances, float *min_obstacle_clearances,
        curandState *rng_states,
        const State start_state, const Control *prev_controls, const Params p,
        const float *ref_xs, const float *ref_ys, const float *ref_yaws, int path_len,
        const float *left_bnd_xs, const float *left_bnd_ys,
        const float *right_bnd_xs, const float *right_bnd_ys,
        const float *, const float* dynamic_obs_x,
        const float* dynamic_obs_y, const float* dynamic_obs_yaw,
        const float* dynamic_obs_semi_major,
        const float* dynamic_obs_semi_minor,
        const std::uint8_t* dynamic_obs_is_dynamic,
        int dynamic_obstacle_count, int dynamic_obstacle_horizon,
        int bnd_len,
        int K, int T, int start_path_idx)
    {
        int k = blockIdx.x * blockDim.x + threadIdx.x;
        if (k >= K) return;

        State x = start_state;
        float total_cost = 0.0f;
        float min_boundary_clearance = 1.0e30f;
        float min_obstacle_clearance = 1.0e30f;
        Control current_action = prev_controls[0]; 
        Control last_u = current_action;

        // White-noise exploration around one nominal can occasionally leave
        // every rollout on the same side of a nearby obstacle. Keep 87.5% of
        // the population completely standard and reserve only one lane per
        // side (2/16 total) for coherent lateral exploration. Do not inject a
        // forced speed reduction: speed remains an MPPI optimization result.
        if (p.rollout_obstacle_ahead) {
            const int avoidance_lane = k & 15;
            if (avoidance_lane == 14) {
                current_action.steer -= 0.24f;
            } else if (avoidance_lane == 15) {
                current_action.steer += 0.24f;
            }
        }
        int local_path_idx = start_path_idx;
        int initial_path_idx = start_path_idx; 
        // Runtime supports only the two servo-lag models. Keeping this array
        // at its actual size avoids hundreds of bytes of unused thread-local
        // storage from the legacy GRU/history implementations.
        float mlp_command_history[12];
        for(int i=0;i<10;++i) mlp_command_history[i]=p.residual_command_history[i];
        mlp_command_history[10]=p.actuator_steer_state;
        mlp_command_history[11]=p.actuator_speed_reference_state;
        
        float x_steer_prev1 = 0.0f, x_steer_prev2 = 0.0f;
        float y_steer_prev1 = 0.0f, y_steer_prev2 = 0.0f;
        float x_accel_prev1 = 0.0f, x_accel_prev2 = 0.0f;
        float y_accel_prev1 = 0.0f, y_accel_prev2 = 0.0f;

        for (int t = 0; t < T; ++t)
        {
            int idx = k * T + t;

            Control u_mean_curr = prev_controls[t];
            Control u_mean_prev = (t == 0) ? prev_controls[0] : prev_controls[t-1];
            
            float mean_delta_steer = u_mean_curr.steer - u_mean_prev.steer;  
            float mean_delta_accel = u_mean_curr.accel - u_mean_prev.accel;  

            // 1. Raw 백색 잡음 생성 (dt를 곱하지 않음)
            float raw_steer_noise = curand_normal(&rng_states[idx]) * p.noise_steer_std;
            float raw_accel_noise = curand_normal(&rng_states[idx]) * p.noise_accel_std;

            // 2. 2차 버터워스 필터링 (IIR 차분 방정식)
            float filtered_steer = p.filter_coeffs.b0 * raw_steer_noise 
                                + p.filter_coeffs.b1 * x_steer_prev1 + p.filter_coeffs.b2 * x_steer_prev2
                                - p.filter_coeffs.a1 * y_steer_prev1 - p.filter_coeffs.a2 * y_steer_prev2;
            
            float filtered_accel = p.filter_coeffs.b0 * raw_accel_noise 
                                + p.filter_coeffs.b1 * x_accel_prev1 + p.filter_coeffs.b2 * x_accel_prev2
                                - p.filter_coeffs.a1 * y_accel_prev1 - p.filter_coeffs.a2 * y_accel_prev2;

            // 3. 레지스터 상태 한 칸씩 시프트 (다음 루프를 위함)
            x_steer_prev2 = x_steer_prev1; x_steer_prev1 = raw_steer_noise;
            y_steer_prev2 = y_steer_prev1; y_steer_prev1 = filtered_steer;

            x_accel_prev2 = x_accel_prev1; x_accel_prev1 = raw_accel_noise;
            y_accel_prev2 = y_accel_prev1; y_accel_prev1 = filtered_accel;

            // 4. 부드러워진 노이즈에 dt를 곱해 최종 변화량 산출
            const float knot_dt=p.model_dt;
            float noise_delta_steer = filtered_steer * knot_dt;
            float noise_delta_accel = filtered_accel * knot_dt;

            current_action.steer += fminf(fmaxf(mean_delta_steer + noise_delta_steer, -p.max_steer_rate * knot_dt), p.max_steer_rate * knot_dt);
            current_action.accel += fminf(fmaxf(mean_delta_accel + noise_delta_accel, -p.max_accel_rate * knot_dt), p.max_accel_rate * knot_dt);

            Control u_clamped = current_action;
            u_clamped.steer = fminf(fmaxf(u_clamped.steer, -p.max_steer), p.max_steer);

            current_action = u_clamped; 

            // The template specialization is the source of truth for the
            // learned correction: DYNAMIC_SERVO_LAG uses the same baseline
            // transition without executing the residual MLP.
            x=update_dynamic_mlp_residual<APPLY_RESIDUAL>(
                x,u_clamped,p,mlp_command_history,true,nullptr);
            states[idx] = x;
            controls[idx] = u_clamped; 

            float min_dist = 1.0e30f;
            if (bnd_len > 0) {
                min_dist = compute_min_boundary_distance(
                    x, ref_xs, ref_ys, ref_yaws, left_bnd_xs, left_bnd_ys,
                    right_bnd_xs, right_bnd_ys, path_len, local_path_idx,
                    &local_path_idx);
                min_boundary_clearance = fminf(min_boundary_clearance, min_dist);
            }

            for (int obstacle_index = 0;
                 obstacle_index < p.num_obstacles && obstacle_index < MAX_OBS;
                 ++obstacle_index) {
                const float dx = x.x - p.obs_x[obstacle_index];
                const float dy = x.y - p.obs_y[obstacle_index];
                min_obstacle_clearance = fminf(
                    min_obstacle_clearance, sqrtf(dx * dx + dy * dy));
            }
            if (dynamic_obstacle_count > 0 && dynamic_obstacle_horizon > 0) {
                const int dynamic_step = min(t, dynamic_obstacle_horizon - 1);
                for (int obstacle_index = 0;
                     obstacle_index < dynamic_obstacle_count; ++obstacle_index) {
                    const int index = obstacle_index * dynamic_obstacle_horizon + dynamic_step;
                    const float dx = x.x - dynamic_obs_x[index];
                    const float dy = x.y - dynamic_obs_y[index];
                    const float distance = sqrtf(dx * dx + dy * dy);
                    float obstacle_radius = 0.0f;
                    if (dynamic_obs_is_dynamic[obstacle_index] && distance > 1.0e-6f) {
                        const float c = fast_cos(dynamic_obs_yaw[index]);
                        const float sn = fast_sin(dynamic_obs_yaw[index]);
                        const float local_x = c * dx + sn * dy;
                        const float local_y = -sn * dx + c * dy;
                        const float major = fmaxf(1.0e-3f, dynamic_obs_semi_major[index]);
                        const float minor = fmaxf(1.0e-3f, dynamic_obs_semi_minor[index]);
                        const float ux = local_x / distance;
                        const float uy = local_y / distance;
                        obstacle_radius = rsqrtf(
                            ux * ux / (major * major) + uy * uy / (minor * minor));
                    } else if (dynamic_obs_is_dynamic[obstacle_index]) {
                        obstacle_radius = fmaxf(dynamic_obs_semi_major[index],
                                                dynamic_obs_semi_minor[index]);
                    }
                    // Clearance summaries share one scalar with legacy static
                    // circles, whose safety threshold is car_radius. Offset an
                    // already-inflated dynamic ellipse so that the common
                    // comparison remains equivalent to distance >= ellipse.
                    const float effective_clearance = distance - obstacle_radius +
                        (dynamic_obs_is_dynamic[obstacle_index] ? p.car_radius : 0.0f);
                    min_obstacle_clearance = fminf(
                        min_obstacle_clearance, effective_clearance);
                }
            }

            if (path_len > 0)
            {
                total_cost += compute_cost_cuda(
                    x,
                    ref_xs, ref_ys, ref_yaws, path_len,
                    u_clamped, last_u, p, min_dist, mlp_command_history[10],
                    t == T - 1,
                    t, dynamic_obs_x, dynamic_obs_y, dynamic_obs_yaw,
                    dynamic_obs_semi_major, dynamic_obs_semi_minor,
                    dynamic_obs_is_dynamic,
                    dynamic_obstacle_count, dynamic_obstacle_horizon,
                    &local_path_idx);
            }

            // 종점 진행도 보상
            if (t == T - 1 && path_len > 0) {
                if(p.objective_mode==MPCC_OBJECTIVE) {
                    float progress_m = compute_progress_distance(
                        start_state, x, initial_path_idx, local_path_idx,
                        ref_xs, ref_ys, ref_yaws, path_len);
                    total_cost -= p.q_progress * progress_m;
                    float terminal_forward_v = x.v * fast_cos(
                        x.yaw - ref_yaws[local_path_idx]);
                    total_cost -= p.q_escape_vel * fmaxf(0.0f, terminal_forward_v);
                } else {
                    total_cost += compute_lmpc_terminal_cost(x,p);
                }
            }

            last_u = u_clamped;
        }
        costs[k] = total_cost;
        min_boundary_clearances[k] = min_boundary_clearance;
        min_obstacle_clearances[k] = min_obstacle_clearance;
    }

    // Re-simulate the control sequence that is actually sent by MPPI.  The
    // displayed minimum-cost sample is not this sequence: MPPI publishes the
    // importance-weighted average of all sampled controls.
    __global__ void weighted_control_rollout_kernel(
        State *states, const Control *controls, State x, const Params p,
        const float *encoded_residual_hidden, int T)
    {
        if (blockIdx.x != 0 || threadIdx.x != 0) return;
        float gru_hidden[RESIDUAL_HIDDEN];
        float command_history[12];
        float vx_history[5];
        for (int i=0;i<RESIDUAL_HIDDEN;++i)
            gru_hidden[i]=encoded_residual_hidden ? encoded_residual_hidden[i] : 0.f;
        for (int i=0;i<10;++i) command_history[i]=p.residual_command_history[i];
        command_history[10]=p.actuator_steer_state;
        command_history[11]=p.actuator_speed_reference_state;
        for(int i=0;i<5;++i)vx_history[i]=p.residual_vx_history[i];
        for (int t=0;t<T;++t) {
            const Control u=controls[t];
            if(p.dynamics_model==KINEMATIC_RESIDUAL)
                x=update_kinematic_residual(x,u,p,gru_hidden);
            else if(p.dynamics_model==KINEMATIC_MLP_RESIDUAL)
                x=update_kinematic_mlp_residual(x,u,p,command_history);
            else if(p.dynamics_model==KINEMATIC_MLP_NO_IMU_RESIDUAL)
                x=update_kinematic_mlp_no_imu_residual(x,u,p,command_history);
            else if(p.dynamics_model==KINEMATIC_NOSLIP_NO_IMU_DIRECT_SPEED)
                x=update_kinematic_noslip_noimu_direct(x,u,p,command_history);
            else if(p.dynamics_model==SLIP_KINEMATIC_WITH_IMU_DIRECT_SPEED)
                x=update_slip_kinematic_with_imu_direct(x,u,p,command_history);
            else if(p.dynamics_model==DYNAMIC_IMU_RECURSIVE)
                x=update_dynamic_imu_recursive(x,u,p,command_history);
            else if(p.dynamics_model==DYNAMIC_SERVO_LAG)
                x=update_dynamic_mlp_residual<false>(
                    x,u,p,command_history,true,nullptr);
            else if(p.dynamics_model==DYNAMIC_MLP_RESIDUAL || p.dynamics_model==DYNAMIC_MLP_RESIDUAL_SERVO_LAG ||
                    p.dynamics_model==DYNAMIC_MLP_RESIDUAL_SERVO_LAG_VX_DELTA_24D)
                x=update_dynamic_mlp_residual<true>(x,u,p,command_history,
                    p.dynamics_model!=DYNAMIC_MLP_RESIDUAL,
                    p.dynamics_model==DYNAMIC_MLP_RESIDUAL_SERVO_LAG_VX_DELTA_24D
                        ? vx_history : nullptr);
            else
                x=update_dynamics(x,u,p);
            states[t]=x;
        }
    }

    // Each horizon knot is independent. Accumulate its importance-weighted
    // control on the GPU and transfer only T controls instead of K*T controls.
    __global__ void weighted_controls_kernel(
        const Control* controls, const float* weights, Control* output,
        int K, int T, float inverse_weight_sum)
    {
        const int t=blockIdx.x*blockDim.x+threadIdx.x;
        if(t>=T)return;
        float steer=0.0f,speed=0.0f;
        for(int k=0;k<K;++k) {
            const float w=weights[k]*inverse_weight_sum;
            const Control u=controls[k*T+t];
            steer=fmaf(w,u.steer,steer);
            speed=fmaf(w,u.accel,speed);
        }
        output[t]={steer,speed};
    }

    // --- Host Functions ---
    __host__ ButterworthCoeffs compute_butterworth_coeffs(float fc, float dt) {
        ButterworthCoeffs coeffs;
        float fs = 1.0f / dt;
        float w0 = tanf(M_PI * fc / fs);
        float K = sqrtf(2.0f) * w0;
        float D = 1.0f + K + w0 * w0;

        coeffs.b0 = (w0 * w0) / D;
        coeffs.b1 = 2.0f * coeffs.b0;
        coeffs.b2 = coeffs.b0;
        coeffs.a1 = 2.0f * (w0 * w0 - 1.0f) / D;
        coeffs.a2 = (1.0f - K + w0 * w0) / D;
        return coeffs;
    }

    MPPISolver::MPPISolver(int K, int T, Params params) : K_(K), T_(T), params_(params) {
        const float knot_dt = uses_40ms_rollout_knot(params_.dynamics_model)
            ? params_.model_dt : params_.dt;
        params_.filter_coeffs = compute_butterworth_coeffs(3.0f, knot_dt);
        h_states_.resize(K * T);
        // A zero steering command is not physically neutral when the
        // identified command mapping contains a bias.  Starting every rollout
        // at zero therefore makes the car curve while the low-pass exploration
        // noise is still too small to compensate.  Warm-start at the command
        // whose effective tire angle is zero and use a modest acceleration so
        // the first horizon can optimize a moving trajectory immediately.
        float neutral_steer = 0.0f;
        if (fabsf(params_.kinematic_steer_scale) > 1.0e-6f) {
            neutral_steer = -params_.kinematic_steer_bias /
                            params_.kinematic_steer_scale;
        }
        neutral_steer = fminf(params_.max_steer,
                              fmaxf(-params_.max_steer, neutral_steer));
        const bool direct_speed=(params_.dynamics_model==KINEMATIC_NOSLIP_NO_IMU_DIRECT_SPEED ||
                                 params_.dynamics_model==SLIP_KINEMATIC_WITH_IMU_DIRECT_SPEED ||
                                 params_.dynamics_model==DYNAMIC_IMU_RECURSIVE ||
                                 params_.dynamics_model==DYNAMIC_MLP_RESIDUAL ||
                                 params_.dynamics_model==DYNAMIC_MLP_RESIDUAL_SERVO_LAG ||
                                 params_.dynamics_model==DYNAMIC_SERVO_LAG ||
                                 params_.dynamics_model==DYNAMIC_MLP_RESIDUAL_SERVO_LAG_VX_DELTA_24D);
        const float initial_accel = direct_speed ? params_.min_speed :
            ((params_.dynamics_model == KINEMATIC) ? 1.0f : 0.0f);
        h_prev_controls_.resize(T, {neutral_steer, initial_accel});
        h_costs_.resize(K);
        h_weights_.resize(K);
        h_min_boundary_clearances_.resize(K);
        h_min_obstacle_clearances_.resize(K);
        allocate_cuda_memory();
    }

    MPPISolver::~MPPISolver() { cleanup_cuda_memory(); }

    void MPPISolver::allocate_cuda_memory() {
        CUDA_CHECK(cudaMalloc(&d_states_, K_ * T_ * sizeof(State)));
        CUDA_CHECK(cudaMalloc(&d_controls_, K_ * T_ * sizeof(Control)));
        CUDA_CHECK(cudaMalloc(&d_prev_controls_, T_ * sizeof(Control)));
        CUDA_CHECK(cudaMalloc(&d_weighted_states_, T_ * sizeof(State)));
        CUDA_CHECK(cudaMalloc(&d_weighted_controls_, T_ * sizeof(Control)));
        CUDA_CHECK(cudaMalloc(&d_costs_, K_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weights_, K_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_min_boundary_clearances_, K_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_min_obstacle_clearances_, K_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rng_states_, K_ * T_ * sizeof(curandState)));
        CUDA_CHECK(cudaMalloc(&d_residual_history_, RESIDUAL_HISTORY*RESIDUAL_FEATURES*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_residual_hidden_, RESIDUAL_HIDDEN*sizeof(float)));
        constexpr int max_dynamic_points = MAX_OBS * MAX_DYNAMIC_OBSTACLE_HORIZON;
        CUDA_CHECK(cudaMalloc(&d_dynamic_obs_x_, max_dynamic_points * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dynamic_obs_y_, max_dynamic_points * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dynamic_obs_yaw_, max_dynamic_points * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dynamic_obs_semi_major_, max_dynamic_points * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dynamic_obs_semi_minor_, max_dynamic_points * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dynamic_obs_is_dynamic_, MAX_OBS * sizeof(std::uint8_t)));
        CUDA_CHECK(cudaMemset(d_residual_history_,0,RESIDUAL_HISTORY*RESIDUAL_FEATURES*sizeof(float)));
        CUDA_CHECK(cudaMemcpyToSymbol(rw_feature_mean,residual_weights::feature_mean,sizeof(residual_weights::feature_mean)));
        CUDA_CHECK(cudaMemcpyToSymbol(rw_feature_std,residual_weights::feature_std,sizeof(residual_weights::feature_std)));
        CUDA_CHECK(cudaMemcpyToSymbol(mlp_feature_mean,mlp_residual_weights::feature_mean,sizeof(mlp_residual_weights::feature_mean)));
        CUDA_CHECK(cudaMemcpyToSymbol(mlp_feature_std,mlp_residual_weights::feature_std,sizeof(mlp_residual_weights::feature_std)));
        CUDA_CHECK(cudaMemcpyToSymbol(mlp_noimu_feature_mean,mlp_no_imu_residual_weights::feature_mean,sizeof(mlp_no_imu_residual_weights::feature_mean)));
        CUDA_CHECK(cudaMemcpyToSymbol(mlp_noimu_feature_std,mlp_no_imu_residual_weights::feature_std,sizeof(mlp_no_imu_residual_weights::feature_std)));
        CUDA_CHECK(cudaMemcpyToSymbol(knd_mean,kinematic_noslip_noimu_direct_weights::feature_mean,sizeof(kinematic_noslip_noimu_direct_weights::feature_mean)));
        CUDA_CHECK(cudaMemcpyToSymbol(knd_std,kinematic_noslip_noimu_direct_weights::feature_std,sizeof(kinematic_noslip_noimu_direct_weights::feature_std)));
        CUDA_CHECK(cudaMemcpyToSymbol(dir_mean,dynamic_imu_recursive_weights::feature_mean,sizeof(dynamic_imu_recursive_weights::feature_mean)));
        CUDA_CHECK(cudaMemcpyToSymbol(dir_std,dynamic_imu_recursive_weights::feature_std,sizeof(dynamic_imu_recursive_weights::feature_std)));
        
        int max_path = 1000;
        CUDA_CHECK(cudaMalloc(&d_ref_xs_, max_path * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ref_ys_, max_path * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ref_yaws_, max_path * sizeof(float)));

        CUDA_CHECK(cudaMalloc(&d_left_bnd_xs_, max_path * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_left_bnd_ys_, max_path * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_right_bnd_xs_, max_path * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_right_bnd_ys_, max_path * sizeof(float)));
        
        int threads = 256;
        int blocks = (K_ * T_ + threads - 1) / threads;
        init_rng_kernel<<<blocks, threads>>>((curandState *)d_rng_states_, 1234UL, K_, T_);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void MPPISolver::cleanup_cuda_memory() {
        cudaFree(d_states_); cudaFree(d_controls_); cudaFree(d_prev_controls_);
        cudaFree(d_weighted_states_); cudaFree(d_weighted_controls_);
        cudaFree(d_costs_); cudaFree(d_weights_);
        cudaFree(d_min_boundary_clearances_);
        cudaFree(d_min_obstacle_clearances_);
        cudaFree(d_rng_states_);
        cudaFree(d_residual_history_); cudaFree(d_residual_hidden_);
        cudaFree(d_dynamic_obs_x_); cudaFree(d_dynamic_obs_y_);
        cudaFree(d_dynamic_obs_yaw_); cudaFree(d_dynamic_obs_semi_major_);
        cudaFree(d_dynamic_obs_semi_minor_); cudaFree(d_dynamic_obs_is_dynamic_);
        cudaFree(d_ref_xs_); cudaFree(d_ref_ys_); cudaFree(d_ref_yaws_);
        cudaFree(d_left_bnd_xs_); cudaFree(d_left_bnd_ys_);
        cudaFree(d_right_bnd_xs_); cudaFree(d_right_bnd_ys_);
    }

    void MPPISolver::update_params(Params p) { 
        params_ = p; 
        const float knot_dt = uses_40ms_rollout_knot(params_.dynamics_model)
            ? params_.model_dt : params_.dt;
        params_.filter_coeffs = compute_butterworth_coeffs(3.0f, knot_dt);
    }   

    void MPPISolver::set_residual_history(const std::vector<float>& features) {
        if(features.size()!=RESIDUAL_HISTORY*RESIDUAL_FEATURES)
            throw std::invalid_argument("residual history must contain 50x11 normalized floats");
        CUDA_CHECK(cudaMemcpy(d_residual_history_,features.data(),features.size()*sizeof(float),cudaMemcpyHostToDevice));
    }

    void MPPISolver::load_residual_weights(const std::string& path) {
        std::ifstream file(path,std::ios::binary);
        if(!file) throw std::runtime_error("cannot open residual weights: "+path);
        std::vector<float> w(62211);
        file.read(reinterpret_cast<char*>(w.data()),w.size()*sizeof(float));
        if(file.gcount()!=static_cast<std::streamsize>(w.size()*sizeof(float)))
            throw std::runtime_error("invalid residual weight file size: "+path);
        size_t o=0;
#define LOAD_RW(symbol,count) CUDA_CHECK(cudaMemcpyToSymbol(symbol,w.data()+o,(count)*sizeof(float))); o+=(count)
        LOAD_RW(rw_enc_w_ih,288*11); LOAD_RW(rw_enc_w_hh,288*96); LOAD_RW(rw_enc_b_ih,288); LOAD_RW(rw_enc_b_hh,288);
        LOAD_RW(rw_dec_w_ih,288*8); LOAD_RW(rw_dec_w_hh,288*96); LOAD_RW(rw_dec_b_ih,288); LOAD_RW(rw_dec_b_hh,288);
        LOAD_RW(rw_head_w,3*96); LOAD_RW(rw_head_b,3);
#undef LOAD_RW
    }

    void MPPISolver::load_mlp_residual_weights(const std::string& path) {
        std::ifstream file(path,std::ios::binary);if(!file)throw std::runtime_error("cannot open MLP weights: "+path);
        std::vector<float>w(3587);file.read(reinterpret_cast<char*>(w.data()),w.size()*sizeof(float));
        if(file.gcount()!=static_cast<std::streamsize>(w.size()*sizeof(float)))throw std::runtime_error("invalid MLP weight file: "+path);
        size_t o=0;
#define LOAD_MLP(symbol,count) CUDA_CHECK(cudaMemcpyToSymbol(symbol,w.data()+o,(count)*sizeof(float)));o+=(count)
        LOAD_MLP(mlp_w1,64*21);LOAD_MLP(mlp_b1,64);LOAD_MLP(mlp_w2,32*64);LOAD_MLP(mlp_b2,32);LOAD_MLP(mlp_w3,3*32);LOAD_MLP(mlp_b3,3);
#undef LOAD_MLP
    }

    void MPPISolver::load_mlp_no_imu_residual_weights(const std::string& path) {
        std::ifstream file(path,std::ios::binary);if(!file)throw std::runtime_error("cannot open no-IMU MLP weights: "+path);
        std::vector<float>w(3395);file.read(reinterpret_cast<char*>(w.data()),w.size()*sizeof(float));
        if(file.gcount()!=static_cast<std::streamsize>(w.size()*sizeof(float)))throw std::runtime_error("invalid no-IMU MLP weight file: "+path);
        size_t o=0;
#define LOAD_MLP_NOIMU(symbol,count) CUDA_CHECK(cudaMemcpyToSymbol(symbol,w.data()+o,(count)*sizeof(float)));o+=(count)
        LOAD_MLP_NOIMU(mlp_noimu_w1,64*18);LOAD_MLP_NOIMU(mlp_noimu_b1,64);LOAD_MLP_NOIMU(mlp_noimu_w2,32*64);LOAD_MLP_NOIMU(mlp_noimu_b2,32);LOAD_MLP_NOIMU(mlp_noimu_w3,3*32);LOAD_MLP_NOIMU(mlp_noimu_b3,3);
#undef LOAD_MLP_NOIMU
    }

    void MPPISolver::load_kinematic_noslip_noimu_direct_weights(const std::string& path) {
        std::ifstream file(path,std::ios::binary);if(!file)throw std::runtime_error("cannot open direct-speed MLP weights: "+path);
        file.seekg(0,std::ios::end);const std::streamoff bytes=file.tellg();file.seekg(0);
        // 16-D files contain 3267 weights followed by mean[16] and std[16].
        const size_t count=static_cast<size_t>(bytes)/sizeof(float);
        if(bytes%sizeof(float)!=0 || (count!=3267 && count!=3267+32))
            throw std::runtime_error("invalid direct-speed MLP weight/normalization file: "+path);
        std::vector<float>w(count);file.read(reinterpret_cast<char*>(w.data()),bytes);
        if(file.gcount()!=bytes)throw std::runtime_error("truncated direct-speed MLP file: "+path);
        size_t o=0;
#define LOAD_KND(symbol,count) CUDA_CHECK(cudaMemcpyToSymbol(symbol,w.data()+o,(count)*sizeof(float)));o+=(count)
        LOAD_KND(knd_w1,64*16);LOAD_KND(knd_b1,64);LOAD_KND(knd_w2,32*64);LOAD_KND(knd_b2,32);LOAD_KND(knd_w3,3*32);LOAD_KND(knd_b3,3);
        if(w.size()==3267+32){LOAD_KND(knd_mean,16);LOAD_KND(knd_std,16);}
#undef LOAD_KND
    }

    void MPPISolver::load_slip_kinematic_with_imu_direct_weights(const std::string& path) {
        std::ifstream file(path,std::ios::binary);if(!file)throw std::runtime_error("cannot open slip direct-speed MLP weights: "+path);
        file.seekg(0,std::ios::end);const std::streamoff bytes=file.tellg();file.seekg(0);
        // 20-D: 3523 network parameters followed by mean[20], std[20].
        const size_t count=static_cast<size_t>(bytes)/sizeof(float);
        if(bytes%sizeof(float)!=0 || count!=3523+40)
            throw std::runtime_error("invalid slip direct-speed MLP weight/normalization file: "+path);
        std::vector<float>w(count);file.read(reinterpret_cast<char*>(w.data()),bytes);
        if(file.gcount()!=bytes)throw std::runtime_error("truncated slip direct-speed MLP file: "+path);
        size_t o=0;
#define LOAD_KSD(symbol,count) CUDA_CHECK(cudaMemcpyToSymbol(symbol,w.data()+o,(count)*sizeof(float)));o+=(count)
        LOAD_KSD(ksd_w1,64*20);LOAD_KSD(ksd_b1,64);LOAD_KSD(ksd_w2,32*64);LOAD_KSD(ksd_b2,32);LOAD_KSD(ksd_w3,3*32);LOAD_KSD(ksd_b3,3);
        LOAD_KSD(ksd_mean,20);LOAD_KSD(ksd_std,20);
#undef LOAD_KSD
    }

    void MPPISolver::load_dynamic_imu_recursive_weights(const std::string& path) {
        std::ifstream file(path,std::ios::binary);if(!file)throw std::runtime_error("cannot open recursive-IMU MLP weights: "+path);
        file.seekg(0,std::ios::end);const std::streamoff bytes=file.tellg();file.seekg(0);
        // Legacy: weights only. New: weights followed by mean[20], std[20].
        const size_t count=static_cast<size_t>(bytes)/sizeof(float);
        if(bytes%sizeof(float)!=0 || (count!=3523 && count!=3523+40))
            throw std::runtime_error("invalid recursive-IMU weight/normalization file: "+path);
        std::vector<float>w(count);file.read(reinterpret_cast<char*>(w.data()),bytes);
        if(file.gcount()!=bytes)throw std::runtime_error("truncated recursive-IMU file: "+path);
        size_t o=0;
#define LOAD_DIR(symbol,count) CUDA_CHECK(cudaMemcpyToSymbol(symbol,w.data()+o,(count)*sizeof(float)));o+=(count)
        LOAD_DIR(dir_w1,64*20);LOAD_DIR(dir_b1,64);LOAD_DIR(dir_w2,32*64);LOAD_DIR(dir_b2,32);LOAD_DIR(dir_w3,3*32);LOAD_DIR(dir_b3,3);
        if(w.size()==3523+40){LOAD_DIR(dir_mean,20);LOAD_DIR(dir_std,20);}
#undef LOAD_DIR
    }

    void MPPISolver::load_dynamic_mlp_residual_weights(const std::string& path) {
        std::ifstream file(path,std::ios::binary);if(!file)throw std::runtime_error("cannot open dynamic residual MLP: "+path);
        file.seekg(0,std::ios::end);const std::streamoff bytes=file.tellg();file.seekg(0);
        const size_t count=static_cast<size_t>(bytes)/sizeof(float);
        constexpr size_t expected=64*22+64+32*64+32+3*32+3+22+22;
        if(bytes%sizeof(float)!=0 || count!=expected)throw std::runtime_error("invalid 22-D IMU dynamic residual MLP file: "+path);
        std::vector<float>w(count);file.read(reinterpret_cast<char*>(w.data()),bytes);
        if(file.gcount()!=bytes)throw std::runtime_error("truncated dynamic residual MLP file: "+path);
        size_t o=0;
#define LOAD_DMR(symbol,count) CUDA_CHECK(cudaMemcpyToSymbol(symbol,w.data()+o,(count)*sizeof(float)));o+=(count)
        LOAD_DMR(dmr_w1,64*22);LOAD_DMR(dmr_b1,64);LOAD_DMR(dmr_w2,32*64);LOAD_DMR(dmr_b2,32);LOAD_DMR(dmr_w3,3*32);LOAD_DMR(dmr_b3,3);LOAD_DMR(dmr_mean,22);LOAD_DMR(dmr_std,22);
#undef LOAD_DMR
    }

    void MPPISolver::load_dynamic_mlp_vx_delta_residual_weights(const std::string& path) {
        std::ifstream file(path,std::ios::binary);
        if(!file)throw std::runtime_error("cannot open 24-D dynamic residual MLP: "+path);
        file.seekg(0,std::ios::end);const std::streamoff bytes=file.tellg();file.seekg(0);
        constexpr size_t count=64*24+64+32*64+32+3*32+3+24+24;
        if(bytes!=static_cast<std::streamoff>(count*sizeof(float)))
            throw std::runtime_error("invalid 24-D dynamic residual MLP file (expected 15308 bytes): "+path);
        std::vector<float>w(count);file.read(reinterpret_cast<char*>(w.data()),bytes);
        if(file.gcount()!=bytes)throw std::runtime_error("truncated 24-D dynamic residual MLP file: "+path);
        size_t o=0;
#define LOAD_DMR24(symbol,count_) CUDA_CHECK(cudaMemcpyToSymbol(symbol,w.data()+o,(count_)*sizeof(float)));o+=(count_)
        LOAD_DMR24(dmr24_w1,64*24);LOAD_DMR24(dmr24_b1,64);
        LOAD_DMR24(dmr24_w2,32*64);LOAD_DMR24(dmr24_b2,32);
        LOAD_DMR24(dmr24_w3,3*32);LOAD_DMR24(dmr24_b3,3);
        LOAD_DMR24(dmr24_mean,24);LOAD_DMR24(dmr24_std,24);
#undef LOAD_DMR24
    }

    State MPPISolver::debug_dynamic_mlp_residual_step(
        const State& state, const Control& control,
        std::array<float, 12>& command_history, bool use_servo_lag)
    {
        float *device_history_input = nullptr;
        float *device_history_output = nullptr;
        State *device_state_output = nullptr;
        CUDA_CHECK(cudaMalloc(&device_history_input, 12 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&device_history_output, 12 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&device_state_output, sizeof(State)));
        CUDA_CHECK(cudaMemcpy(device_history_input, command_history.data(),
                              12 * sizeof(float), cudaMemcpyHostToDevice));
        debug_dynamic_mlp_residual_step_kernel<<<1, 1>>>(
            state, control, params_, device_history_input,
            device_state_output, device_history_output, use_servo_lag);
        CUDA_CHECK(cudaDeviceSynchronize());
        State output{};
        CUDA_CHECK(cudaMemcpy(&output, device_state_output, sizeof(State),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(command_history.data(), device_history_output,
                              12 * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(device_history_input);
        cudaFree(device_history_output);
        cudaFree(device_state_output);
        return output;
    }

    State MPPISolver::debug_dynamic_mlp_vx_delta_residual_step(
        const State& state,const Control& control,std::array<float,12>& command_history,
        std::array<float,5>& vx_history)
    {
        float *dh=nullptr,*dho=nullptr,*dv=nullptr,*dvo=nullptr;State *ds=nullptr;
        CUDA_CHECK(cudaMalloc(&dh,12*sizeof(float)));CUDA_CHECK(cudaMalloc(&dho,12*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dv,5*sizeof(float)));CUDA_CHECK(cudaMalloc(&dvo,5*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ds,sizeof(State)));
        CUDA_CHECK(cudaMemcpy(dh,command_history.data(),12*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dv,vx_history.data(),5*sizeof(float),cudaMemcpyHostToDevice));
        debug_dynamic_mlp_vx_delta_step_kernel<<<1,1>>>(state,control,params_,dh,dv,ds,dho,dvo);
        CUDA_CHECK(cudaDeviceSynchronize());State output{};
        CUDA_CHECK(cudaMemcpy(&output,ds,sizeof(State),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(command_history.data(),dho,12*sizeof(float),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vx_history.data(),dvo,5*sizeof(float),cudaMemcpyDeviceToHost));
        cudaFree(dh);cudaFree(dho);cudaFree(dv);cudaFree(dvo);cudaFree(ds);return output;
    }

    void MPPISolver::set_reference_path(const std::vector<float> &xs, const std::vector<float> &ys,
                                        const std::vector<float> &yaws) {
        h_ref_xs_ = xs; h_ref_ys_ = ys; h_ref_yaws_ = yaws;
        ref_path_len_ = xs.size();
        if (ref_path_len_ > 1000) ref_path_len_ = 1000;
        if (ref_path_len_ > 0) {
            CUDA_CHECK(cudaMemcpy(d_ref_xs_, xs.data(), ref_path_len_ * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ref_ys_, ys.data(), ref_path_len_ * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ref_yaws_, yaws.data(), ref_path_len_ * sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    void MPPISolver::set_boundaries(const std::vector<float>& left_xs, const std::vector<float>& left_ys,
                                    const std::vector<float>& right_xs, const std::vector<float>& right_ys) {
        h_left_bnd_xs_ = left_xs;
        h_left_bnd_ys_ = left_ys;
        h_right_bnd_xs_ = right_xs;
        h_right_bnd_ys_ = right_ys;
        bnd_len_ = left_xs.size();
        if (bnd_len_ > 1000) bnd_len_ = 1000; 
        if (bnd_len_ > 0) {
            CUDA_CHECK(cudaMemcpy(d_left_bnd_xs_, left_xs.data(), bnd_len_ * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_left_bnd_ys_, left_ys.data(), bnd_len_ * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_right_bnd_xs_, right_xs.data(), bnd_len_ * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_right_bnd_ys_, right_ys.data(), bnd_len_ * sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    void MPPISolver::set_dynamic_obstacles(
        const std::vector<float>& xs, const std::vector<float>& ys,
        const std::vector<float>& yaws,
        const std::vector<float>& semi_major,
        const std::vector<float>& semi_minor,
        const std::vector<bool>& is_dynamic,
        int obstacle_count, int horizon) {
        dynamic_obstacle_count_ = std::max(0, std::min(obstacle_count, MAX_OBS));
        dynamic_obstacle_horizon_ = std::max(0,
            std::min(horizon, MAX_DYNAMIC_OBSTACLE_HORIZON));
        const std::size_t count = static_cast<std::size_t>(
            dynamic_obstacle_count_ * dynamic_obstacle_horizon_);
        if (count == 0 || xs.size() < count || ys.size() < count ||
            yaws.size() < count || semi_major.size() < count ||
            semi_minor.size() < count || is_dynamic.size() < static_cast<std::size_t>(dynamic_obstacle_count_)) {
            dynamic_obstacle_count_ = 0;
            dynamic_obstacle_horizon_ = 0;
            h_dynamic_obs_x_.clear(); h_dynamic_obs_y_.clear();
            h_dynamic_obs_yaw_.clear(); h_dynamic_obs_semi_major_.clear();
            h_dynamic_obs_semi_minor_.clear(); h_dynamic_obs_is_dynamic_.clear();
            return;
        }
        h_dynamic_obs_x_.assign(xs.begin(), xs.begin() + count);
        h_dynamic_obs_y_.assign(ys.begin(), ys.begin() + count);
        h_dynamic_obs_yaw_.assign(yaws.begin(), yaws.begin() + count);
        h_dynamic_obs_semi_major_.assign(semi_major.begin(), semi_major.begin() + count);
        h_dynamic_obs_semi_minor_.assign(semi_minor.begin(), semi_minor.begin() + count);
        h_dynamic_obs_is_dynamic_.assign(is_dynamic.begin(),
            is_dynamic.begin() + dynamic_obstacle_count_);
        std::vector<std::uint8_t> dynamic_flags(dynamic_obstacle_count_);
        for (int i = 0; i < dynamic_obstacle_count_; ++i)
            dynamic_flags[i] = h_dynamic_obs_is_dynamic_[i] ? 1u : 0u;
        CUDA_CHECK(cudaMemcpy(d_dynamic_obs_x_, h_dynamic_obs_x_.data(),
                              count * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dynamic_obs_y_, h_dynamic_obs_y_.data(),
                              count * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dynamic_obs_yaw_, h_dynamic_obs_yaw_.data(),
                              count * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dynamic_obs_semi_major_, h_dynamic_obs_semi_major_.data(),
                              count * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dynamic_obs_semi_minor_, h_dynamic_obs_semi_minor_.data(),
                              count * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dynamic_obs_is_dynamic_, dynamic_flags.data(),
                              dynamic_flags.size() * sizeof(std::uint8_t), cudaMemcpyHostToDevice));
    }

    Control MPPISolver::solve(const State &current_state) {
        const bool first_direct_speed_solve=!direct_speed_warm_start_initialized_;
        params_.rollout_obstacle_ahead=false;
        const float start_cos=std::cos(current_state.yaw);
        const float start_sin=std::sin(current_state.yaw);
        for(int i=0;i<params_.num_obstacles;++i) {
            const float dx=params_.obs_x[i]-current_state.x;
            const float dy=params_.obs_y[i]-current_state.y;
            if(dx*start_cos+dy*start_sin>-0.25f && dx*dx+dy*dy<36.0f) {
                params_.rollout_obstacle_ahead=true;
                break;
            }
        }
        if (!params_.rollout_obstacle_ahead && dynamic_obstacle_count_ > 0 &&
            dynamic_obstacle_horizon_ > 0) {
            for (int i = 0; i < dynamic_obstacle_count_; ++i) {
                const int index = i * dynamic_obstacle_horizon_;
                const float dx = h_dynamic_obs_x_[index] - current_state.x;
                const float dy = h_dynamic_obs_y_[index] - current_state.y;
                if (dx * start_cos + dy * start_sin > -0.25f &&
                    dx * dx + dy * dy < 36.0f) {
                    params_.rollout_obstacle_ahead = true;
                    break;
                }
            }
        }
        int start_path_idx = 0;
        if (!h_ref_xs_.empty()) {
            float min_dist_sq = 1e9f;
            for (int i = 0; i < (int)h_ref_xs_.size(); ++i) {
                float dx = current_state.x - h_ref_xs_[i];
                float dy = current_state.y - h_ref_ys_[i];
                float d_sq = dx*dx + dy*dy;
                if (d_sq < min_dist_sq) { min_dist_sq = d_sq; start_path_idx = i; }
            }
        }

        if (first_direct_speed_solve) {
            // The old first-solve nominal accelerated along zero steering. On a
            // corner this makes most samples start from an off-track horizon.
            // Seed the nominal with centerline curvature; MPPI remains free to
            // move away from the centerline after the first optimization.
            float speed=fminf(params_.max_speed,fmaxf(0.0f,current_state.v));
            const float knot_dt=uses_40ms_rollout_knot(params_.dynamics_model)
                ? params_.model_dt : params_.dt;
            const float speed_step=params_.max_accel_rate*knot_dt;
            int path_idx=start_path_idx;
            float distance_to_next=0.0f;
            for (int t=0;t<T_;++t) {
                // Curvature seeds steering feasibility; keep the same speed
                // envelope as subsequent solves so initialization has no
                // artificial lap-time penalty.
                speed=fminf(params_.max_speed,speed+speed_step);
                h_prev_controls_[t].accel=speed;
                if (ref_path_len_ > 2 && h_ref_yaws_.size() >= static_cast<size_t>(ref_path_len_)) {
                    distance_to_next += fmaxf(speed, params_.min_speed)*knot_dt;
                    while (distance_to_next > 0.0f) {
                        const int next=(path_idx+1)%ref_path_len_;
                        const float dx=h_ref_xs_[next]-h_ref_xs_[path_idx];
                        const float dy=h_ref_ys_[next]-h_ref_ys_[path_idx];
                        const float ds=fmaxf(1.0e-3f,sqrtf(dx*dx+dy*dy));
                        if (distance_to_next < ds) break;
                        distance_to_next -= ds;
                        path_idx=next;
                    }
                    // Use a wide geometric stencil. Per-waypoint heading
                    // differences in measured centerlines are too noisy for a
                    // feed-forward steering seed and can alternate in sign.
                    constexpr int curvature_window=8;
                    const int before=(path_idx+ref_path_len_-curvature_window)%ref_path_len_;
                    const int after=(path_idx+curvature_window)%ref_path_len_;
                    const float before_dx=h_ref_xs_[path_idx]-h_ref_xs_[before];
                    const float before_dy=h_ref_ys_[path_idx]-h_ref_ys_[before];
                    const float after_dx=h_ref_xs_[after]-h_ref_xs_[path_idx];
                    const float after_dy=h_ref_ys_[after]-h_ref_ys_[path_idx];
                    const float before_heading=atan2f(before_dy,before_dx);
                    const float after_heading=atan2f(after_dy,after_dx);
                    const float stencil_length=fmaxf(1.0e-3f,0.5f*(
                        hypotf(before_dx,before_dy)+hypotf(after_dx,after_dy)));
                    const float curvature=angle_normalize(
                        after_heading-before_heading)/stencil_length;
                    const float target_tire_steer=atanf((params_.l_f+params_.l_r)*curvature);
                    const float scale=fabsf(params_.kinematic_steer_scale)>1.0e-6f
                        ? params_.kinematic_steer_scale : 1.0f;
                    h_prev_controls_[t].steer=fminf(params_.max_steer,
                        fmaxf(-params_.max_steer,
                            (target_tire_steer-params_.kinematic_steer_bias)/scale));
                }
            }
            direct_speed_warm_start_initialized_=true;
        }
        CUDA_CHECK(cudaMemcpy(d_prev_controls_, h_prev_controls_.data(), T_ * sizeof(Control), cudaMemcpyHostToDevice));

        int threadsPerBlock = 128;
        int blocksPerGrid = (K_ + threadsPerBlock - 1) / threadsPerBlock;
        if(params_.dynamics_model==KINEMATIC_RESIDUAL)
            encode_residual_history<<<1,1>>>(d_residual_history_,d_residual_hidden_);

        if (params_.dynamics_model==DYNAMIC_MLP_RESIDUAL_SERVO_LAG) {
            rollout_kernel<true><<<blocksPerGrid, threadsPerBlock>>>(
                d_states_, d_controls_, d_costs_, d_min_boundary_clearances_,
                d_min_obstacle_clearances_, (curandState *)d_rng_states_,
                current_state, d_prev_controls_, params_,
                d_ref_xs_, d_ref_ys_, d_ref_yaws_, ref_path_len_,
                d_left_bnd_xs_, d_left_bnd_ys_, d_right_bnd_xs_, d_right_bnd_ys_,
                d_residual_hidden_, d_dynamic_obs_x_, d_dynamic_obs_y_,
                d_dynamic_obs_yaw_, d_dynamic_obs_semi_major_,
                d_dynamic_obs_semi_minor_, d_dynamic_obs_is_dynamic_,
                dynamic_obstacle_count_,
                dynamic_obstacle_horizon_, bnd_len_, K_, T_, start_path_idx);
        } else {
            rollout_kernel<false><<<blocksPerGrid, threadsPerBlock>>>(
                d_states_, d_controls_, d_costs_, d_min_boundary_clearances_,
                d_min_obstacle_clearances_, (curandState *)d_rng_states_,
                current_state, d_prev_controls_, params_,
                d_ref_xs_, d_ref_ys_, d_ref_yaws_, ref_path_len_,
                d_left_bnd_xs_, d_left_bnd_ys_, d_right_bnd_xs_, d_right_bnd_ys_,
                d_residual_hidden_, d_dynamic_obs_x_, d_dynamic_obs_y_,
                d_dynamic_obs_yaw_, d_dynamic_obs_semi_major_,
                d_dynamic_obs_semi_minor_, d_dynamic_obs_is_dynamic_,
                dynamic_obstacle_count_,
                dynamic_obstacle_horizon_, bnd_len_, K_, T_, start_path_idx);
        }
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_costs_.data(), d_costs_, K_ * sizeof(float), cudaMemcpyDeviceToHost));
        if (params_.weighted_trajectory_safety_enabled) {
            CUDA_CHECK(cudaMemcpy(h_min_boundary_clearances_.data(),
                                  d_min_boundary_clearances_,
                                  K_ * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_min_obstacle_clearances_.data(),
                                  d_min_obstacle_clearances_,
                                  K_ * sizeof(float), cudaMemcpyDeviceToHost));
        }
        
        if (params_.visualize_candidates) {
            CUDA_CHECK(cudaMemcpy(h_states_.data(), d_states_, K_ * T_ * sizeof(State), cudaMemcpyDeviceToHost));
        }
        return compute_optimal_control(current_state);
    }

    float MPPISolver::trajectory_min_boundary_clearance(
        const std::vector<State>& trajectory) const {
        const size_t n=std::min({h_ref_xs_.size(),h_ref_ys_.size(),
            h_left_bnd_xs_.size(),h_left_bnd_ys_.size(),
            h_right_bnd_xs_.size(),h_right_bnd_ys_.size()});
        if(n==0 || trajectory.empty()) return std::numeric_limits<float>::infinity();
        float minimum=std::numeric_limits<float>::infinity();
        for(const State& state:trajectory) {
            size_t nearest=0;
            float nearest_distance_sq=std::numeric_limits<float>::infinity();
            for(size_t i=0;i<n;++i) {
                const float dx=state.x-h_ref_xs_[i];
                const float dy=state.y-h_ref_ys_[i];
                const float distance_sq=dx*dx+dy*dy;
                if(distance_sq<nearest_distance_sq) {
                    nearest_distance_sq=distance_sq;
                    nearest=i;
                }
            }
            const float yaw=h_ref_yaws_.size()>nearest?h_ref_yaws_[nearest]:0.0f;
            const float lateral_error=-(state.x-h_ref_xs_[nearest])*sinf(yaw)
                                      +(state.y-h_ref_ys_[nearest])*cosf(yaw);
            const float left_width=hypotf(h_left_bnd_xs_[nearest]-h_ref_xs_[nearest],
                                          h_left_bnd_ys_[nearest]-h_ref_ys_[nearest]);
            const float right_width=hypotf(h_right_bnd_xs_[nearest]-h_ref_xs_[nearest],
                                           h_right_bnd_ys_[nearest]-h_ref_ys_[nearest]);
            minimum=std::min(minimum,std::min(left_width-lateral_error,
                                              right_width+lateral_error));
        }
        return minimum;
    }

    float MPPISolver::trajectory_min_obstacle_clearance(
        const std::vector<State>& trajectory) const {
        if((params_.num_obstacles<=0 || trajectory.empty()) &&
           (dynamic_obstacle_count_ <= 0 || dynamic_obstacle_horizon_ <= 0))
            return std::numeric_limits<float>::infinity();
        float minimum=std::numeric_limits<float>::infinity();
        for(size_t step=0; step<trajectory.size(); ++step) {
            const State& state=trajectory[step];
            for(int i=0;i<params_.num_obstacles;++i) {
                const float dx=state.x-params_.obs_x[i];
                const float dy=state.y-params_.obs_y[i];
                minimum=std::min(minimum,std::sqrt(dx*dx+dy*dy));
            }
            if(dynamic_obstacle_count_ > 0 && dynamic_obstacle_horizon_ > 0) {
                const int dynamic_step=std::min(static_cast<int>(step),
                                                dynamic_obstacle_horizon_-1);
                for(int i=0;i<dynamic_obstacle_count_;++i) {
                    const int index=i*dynamic_obstacle_horizon_+dynamic_step;
                    const float dx=state.x-h_dynamic_obs_x_[index];
                    const float dy=state.y-h_dynamic_obs_y_[index];
                    const float distance=std::sqrt(dx*dx+dy*dy);
                    float obstacle_radius=0.0f;
                    if(h_dynamic_obs_is_dynamic_[i] && distance>1.0e-6f) {
                        const float c=std::cos(h_dynamic_obs_yaw_[index]);
                        const float sn=std::sin(h_dynamic_obs_yaw_[index]);
                        const float local_x=c*dx+sn*dy;
                        const float local_y=-sn*dx+c*dy;
                        const float ux=local_x/distance,uy=local_y/distance;
                        const float major=std::max(1.0e-3f,h_dynamic_obs_semi_major_[index]);
                        const float minor=std::max(1.0e-3f,h_dynamic_obs_semi_minor_[index]);
                        obstacle_radius=1.0f/std::sqrt(
                            ux*ux/(major*major)+uy*uy/(minor*minor));
                    } else if(h_dynamic_obs_is_dynamic_[i]) {
                        obstacle_radius=std::max(h_dynamic_obs_semi_major_[index],
                                                 h_dynamic_obs_semi_minor_[index]);
                    }
                    minimum=std::min(minimum,distance-obstacle_radius+
                        (h_dynamic_obs_is_dynamic_[i] ? params_.car_radius : 0.0f));
                }
            }
        }
        return minimum;
    }

    Control MPPISolver::compute_optimal_control(const State &current_state) {
        const bool phase_aware = uses_40ms_rollout_knot(params_.dynamics_model);
        const bool advance_warm_start = !phase_aware || model_knot_phase_==1;
        if(phase_aware)model_knot_phase_=(model_knot_phase_+1)&1;
        auto min_it = std::min_element(h_costs_.begin(), h_costs_.end());
        float min_cost = *min_it;
        best_k_ = static_cast<int>(std::distance(h_costs_.begin(), min_it));

        // A large finite cost is still a valid soft-constrained solution.
        // Only non-finite rollouts may reuse the previous horizon.
        if (!std::isfinite(min_cost)) {
            const std::vector<Control> fallback_controls = h_prev_controls_;
            optimal_controls_ = fallback_controls;
            weighted_control_trajectory_.resize(T_);
            CUDA_CHECK(cudaMemcpy(d_weighted_controls_, fallback_controls.data(),
                                  T_*sizeof(Control), cudaMemcpyHostToDevice));
            weighted_control_rollout_kernel<<<1,1>>>(
                d_weighted_states_,d_weighted_controls_,current_state,params_,
                d_residual_hidden_,T_);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(weighted_control_trajectory_.data(),d_weighted_states_,
                                  T_*sizeof(State),cudaMemcpyDeviceToHost));
            best_trajectory_ = weighted_control_trajectory_;

            if(advance_warm_start) {
                for (int t = 0; t < T_ - 1; ++t)h_prev_controls_[t] = fallback_controls[t + 1];
                h_prev_controls_[T_ - 1] = fallback_controls[T_ - 1];
            }
            return fallback_controls[0];
        }
        for (int k = 0; k < K_; ++k) {
            if (std::isnan(h_costs_[k])) {
                h_costs_[k] = 1.0e8f; // NaN 발생 시 최악의 비용으로 간주하여 도태시킴
            }
        }

        float lambda = params_.lambda; 
        float sum_weights = 0.0f;
        for (int k = 0; k < K_; ++k) {
            if (std::isinf(h_costs_[k])) {
                h_weights_[k] = 0.0f; 
            } else {
                h_weights_[k] = fast_exp(-(h_costs_[k] - min_cost) / lambda);
            }
            sum_weights += h_weights_[k];
        }
        if (sum_weights < 1e-6) sum_weights = 1e-6;

        std::vector<Control> weighted_controls(T_);
        CUDA_CHECK(cudaMemcpy(d_weights_,h_weights_.data(),K_*sizeof(float),
                              cudaMemcpyHostToDevice));
        constexpr int weighted_threads=128;
        weighted_controls_kernel<<<(T_+weighted_threads-1)/weighted_threads,
            weighted_threads>>>(d_controls_,d_weights_,d_weighted_controls_,
                                 K_,T_,1.0f/sum_weights);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(weighted_controls.data(),d_weighted_controls_,
                              T_*sizeof(Control),cudaMemcpyDeviceToHost));
        
        optimal_controls_ = weighted_controls; 
        Control output = weighted_controls[0];

        weighted_control_trajectory_.resize(T_);
        CUDA_CHECK(cudaMemcpy(d_weighted_controls_, weighted_controls.data(),
                              T_ * sizeof(Control), cudaMemcpyHostToDevice));
        weighted_control_rollout_kernel<<<1,1>>>(
            d_weighted_states_, d_weighted_controls_, current_state, params_,
            d_residual_hidden_, T_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(weighted_control_trajectory_.data(), d_weighted_states_,
                              T_ * sizeof(State), cudaMemcpyDeviceToHost));

        // Safe samples from opposite modes can average into an unsafe path.
        // Check the rollout of the controls that will actually be published.
        if(params_.weighted_trajectory_safety_enabled) {
            const bool weighted_obstacle_unsafe =
                trajectory_min_obstacle_clearance(weighted_control_trajectory_)
                    < params_.car_radius;
            const bool weighted_boundary_unsafe =
                trajectory_min_boundary_clearance(weighted_control_trajectory_)
                    < 0.0f;
            if (weighted_obstacle_unsafe || weighted_boundary_unsafe) {
                // The rollout kernel already computed the minimum clearance
                // for every candidate. Use those K summaries instead of
                // copying and rescanning K*T states on the CPU (which was the
                // source of the 100--300 ms safety spikes on the vehicle
                // computer).
                std::vector<int> candidates(K_);
                for(int k=0;k<K_;++k)candidates[k]=k;
                std::sort(candidates.begin(),candidates.end(),[this](int lhs,int rhs) {
                    return h_costs_[lhs]<h_costs_[rhs];
                });
                int safe_candidate=-1;
                for(int k:candidates) {
                    if(h_min_obstacle_clearances_[k] >= params_.car_radius &&
                       h_min_boundary_clearances_[k] >= 0.0f) {
                        safe_candidate=k;
                        break;
                    }
                }
                if(safe_candidate>=0) {
                best_k_=safe_candidate;
                CUDA_CHECK(cudaMemcpy(
                    weighted_controls.data(),
                    d_controls_ + static_cast<size_t>(safe_candidate) * T_,
                    T_ * sizeof(Control), cudaMemcpyDeviceToHost));
                optimal_controls_=weighted_controls;
                output=weighted_controls[0];
                CUDA_CHECK(cudaMemcpy(d_weighted_controls_,weighted_controls.data(),
                                      T_*sizeof(Control),cudaMemcpyHostToDevice));
                weighted_control_rollout_kernel<<<1,1>>>(
                    d_weighted_states_,d_weighted_controls_,current_state,params_,
                    d_residual_hidden_,T_);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(weighted_control_trajectory_.data(),d_weighted_states_,
                                      T_*sizeof(State),cudaMemcpyDeviceToHost));
                } else {
                // Replaying the previous steering when every candidate is
                // unsafe can lock the vehicle into an off-track circle. Build
                // a low-speed geometric recovery sequence toward a centerline
                // look-ahead point so the next solve starts from a recoverable
                // state instead.
                size_t nearest_index=0;
                float nearest_distance_sq=std::numeric_limits<float>::infinity();
                for(size_t index=0;index<h_ref_xs_.size();++index) {
                    const float dx=current_state.x-h_ref_xs_[index];
                    const float dy=current_state.y-h_ref_ys_[index];
                    const float distance_sq=dx*dx+dy*dy;
                    if(distance_sq<nearest_distance_sq) {
                        nearest_distance_sq=distance_sq;
                        nearest_index=index;
                    }
                }
                const size_t lookahead_index=h_ref_xs_.empty()?0:
                    (nearest_index+8)%h_ref_xs_.size();
                const float desired_heading=h_ref_xs_.empty()?current_state.yaw:
                    std::atan2(h_ref_ys_[lookahead_index]-current_state.y,
                               h_ref_xs_[lookahead_index]-current_state.x);
                const float heading_error=std::atan2(
                    std::sin(desired_heading-current_state.yaw),
                    std::cos(desired_heading-current_state.yaw));
                constexpr float max_wheel_steer=.55f;
                float desired_applied_steer=std::min(max_wheel_steer,
                    std::max(-max_wheel_steer,0.8f*heading_error));
                const float steer_scale=std::fabs(params_.kinematic_steer_scale)>1.0e-6f
                    ?params_.kinematic_steer_scale:1.0f;
                const float recovery_steer=std::min(params_.max_steer,
                    std::max(-params_.max_steer,
                        (desired_applied_steer-params_.kinematic_steer_bias)/steer_scale));
                weighted_controls.assign(T_,{
                    recovery_steer,
                    fminf(params_.min_speed,fmaxf(0.0f,current_state.v))});
                optimal_controls_=weighted_controls;
                output=weighted_controls[0];
                CUDA_CHECK(cudaMemcpy(d_weighted_controls_,weighted_controls.data(),
                                      T_*sizeof(Control),cudaMemcpyHostToDevice));
                weighted_control_rollout_kernel<<<1,1>>>(
                    d_weighted_states_,d_weighted_controls_,current_state,params_,
                    d_residual_hidden_,T_);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(weighted_control_trajectory_.data(),d_weighted_states_,
                                      T_*sizeof(State),cudaMemcpyDeviceToHost));
                }
            }
        }

        if(advance_warm_start) {
            for (int t = 0; t < T_ - 1; ++t) h_prev_controls_[t] = weighted_controls[t + 1];
            h_prev_controls_[T_ - 1] = weighted_controls[T_ - 1];
        } else {
            h_prev_controls_=weighted_controls;
        }

        // The weighted sequence was already rolled out on CUDA above. Reuse
        // it instead of running a second, legacy host dynamics loop.
        best_trajectory_ = weighted_control_trajectory_;
        return output;
    }

    const std::vector<State> &MPPISolver::get_generated_trajectories() const { return h_states_; }
    const std::vector<State> &MPPISolver::get_best_trajectory() const { return best_trajectory_; }
    const std::vector<State> &MPPISolver::get_weighted_control_trajectory() const { return weighted_control_trajectory_; }
    int MPPISolver::get_best_k() const { return best_k_; }
    const std::vector<Control>& MPPISolver::get_optimal_controls() const { return optimal_controls_; }
    const std::vector<float>& MPPISolver::get_costs() const { return h_costs_; }
    int MPPISolver::get_K() const { return K_; }
    int MPPISolver::get_T() const { return T_; }
}
