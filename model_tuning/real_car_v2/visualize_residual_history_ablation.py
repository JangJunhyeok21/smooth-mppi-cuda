#!/usr/bin/env python3
"""Plot causal residual-history ablation results without retraining."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RESULT=ROOT/'model_tuning/results/residual_history_ablation/results.json'
OUTPUT=RESULT.with_name('residual_history_feature_ablation.png')

def main():
 data=json.loads(RESULT.read_text());names=data['ranking'];items=[data['variants'][name] for name in names];y=np.arange(len(names));fig,axes=plt.subplots(1,3,figsize=(18,7),sharey=True)
 fields=(('trajectory_m','Trajectory error [m]'),('yaw_rad','Yaw error [rad]'),('yaw_rate_rps','Yaw-rate error [rad/s]'))
 for axis,(field,title) in zip(axes,fields):
  mean=[item['test_rollout'][field]['mean'] for item in items];p95=[item['test_rollout'][field]['p95'] for item in items];axis.barh(y,np.asarray(p95),color='tab:orange',alpha=.55,label='test p95');axis.barh(y,np.asarray(mean),color='tab:blue',label='test mean');axis.set_title(title);axis.grid(axis='x',alpha=.25);axis.legend()
 axes[0].set_yticks(y,labels=names);axes[0].invert_yaxis();fig.suptitle('Causal history-feature ablation (1.2 s free rollout)');fig.tight_layout();fig.savefig(OUTPUT,dpi=180,bbox_inches='tight');print(OUTPUT)
if __name__=='__main__':main()
