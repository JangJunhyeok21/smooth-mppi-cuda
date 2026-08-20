#!/usr/bin/env python3
"""F5로 simulator GRU closed-loop MPPI 튜닝 결과를 시각화한다."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2];SEARCH=ROOT/'model_tuning/results/map1_simulator_gru_controller_tuning';CONFIRM=ROOT/'model_tuning/results/map1_simulator_gru_controller_confirmation'
def main():
 rows=json.loads((SEARCH/'results.json').read_text());confirmed=json.loads((CONFIRM/'results.json').read_text());fig,axes=plt.subplots(1,2,figsize=(13,5));names=[r['variant'] for r in rows];values=[r['seconds_per_lap'] if r['status']=='laps_complete' else np.nan for r in rows];colors=['tab:green' if r['status']=='laps_complete' else 'tab:red' for r in rows];axes[0].bar(names,values,color=colors);axes[0].set_ylabel('seconds/lap');axes[0].set_title('1-lap search (red = collision/timeout)');axes[0].tick_params(axis='x',rotation=18);axes[0].grid(axis='y',alpha=.3)
 names=[r['variant'] for r in confirmed];axes[1].bar(names,[r['seconds_per_lap'] for r in confirmed],color=('tab:green','tab:blue'));axes[1].set_ylabel('seconds/lap');axes[1].set_title('2-lap confirmation');axes[1].grid(axis='y',alpha=.3)
 fig.tight_layout();out=CONFIRM/'controller_tuning_summary.png';fig.savefig(out,dpi=180);print(out);plt.show()
if __name__=='__main__':main()
