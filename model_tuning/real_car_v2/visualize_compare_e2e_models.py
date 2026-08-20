#!/usr/bin/env python3
"""F5로 deployed/new residual/E2E의 1.2 s rollout을 정량 비교한다."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
COMPARE=ROOT/'model_tuning/results/compare_0817_0820_inertial_ekf_bias'
E2E=ROOT/'model_tuning/results/e2e_0817_0820_inertial_ekf_bias_seed31'
def main():
 paths={'deployed residual':COMPARE/'deployed_lateral_only.json','new EKF residual':COMPARE/'new_lateral_only.json','E2E':E2E/'rollout_60step_metrics.json'}
 metrics={name:json.loads(path.read_text())['test_aggressive'] for name,path in paths.items()}
 report={'horizon_s':1.2,'split':'test_aggressive','models':metrics};out=COMPARE/'e2e_quantitative_comparison.json';out.write_text(json.dumps(report,indent=2)+'\n')
 fig,axes=plt.subplots(1,2,figsize=(14,5.5));names=list(metrics);x=np.arange(len(names));width=.24
 for j,key in enumerate(('mean','p95','max')):axes[0].bar(x+(j-1)*width,[metrics[n]['trajectory_m'][key] for n in names],width,label=key)
 axes[0].set_xticks(x,names);axes[0].set_ylabel('trajectory endpoint error [m]');axes[0].set_title('1.2 s trajectory error');axes[0].grid(axis='y',alpha=.3);axes[0].legend()
 state_keys=('vx_mps','vy_mps','yaw_rate_rps');labels=('vx','vy','yaw-rate');sx=np.arange(3)
 for j,name in enumerate(names):axes[1].bar(sx+(j-1)*width,[metrics[name][k]['mean'] for k in state_keys],width,label=name)
 axes[1].set_xticks(sx,labels);axes[1].set_ylabel('endpoint absolute error');axes[1].set_title('1.2 s endpoint state mean error');axes[1].grid(axis='y',alpha=.3);axes[1].legend()
 fig.suptitle('0817-0820 held-out model comparison');fig.tight_layout();png=COMPARE/'e2e_quantitative_comparison.png';fig.savefig(png,dpi=180);print(png);print(out);plt.show()
if __name__=='__main__':main()
