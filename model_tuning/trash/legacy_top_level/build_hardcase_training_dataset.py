#!/usr/bin/env python3
"""Combine all freshly extracted /ackermann_cmd bags with a bag-disjoint split."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCES = [
    # train
    (ROOT / "model_tuning/data/extracted_bags/rosbag2_2026_08_07-19_13_58.npz", 0, "0807 19:13:58"),
    (ROOT / "model_tuning/data/extracted_bags/rosbag2_2026_08_08-16_54_33.npz", 0, "0808 16:54:33"),
    (ROOT / "model_tuning/data/extracted_bags/rosbag2_2026_08_08-20_19_06.npz", 0, "0808 20:19:06"),
    (ROOT / "model_tuning/data/extracted_bags/rosbag2_2026_08_08-20_20_34.npz", 0, "0808 20:20:34"),
    (ROOT / "model_tuning/data/extracted_bags/rosbag2_2026_08_10-21_46_44.npz", 0, "0810 21:46:44"),

    # Held-out physical bags. Keep complete bags out of training.
    (ROOT / "model_tuning/data/extracted_bags/rosbag2_2026_08_08-20_25_26.npz", 1, "0808 20:25:26"),
    (ROOT / "model_tuning/data/extracted_bags/rosbag2_2026_08_08-22_10_38.npz", 1, "0808 22:10:38"),
    (ROOT / "model_tuning/data/extracted_bags/rosbag2_2026_08_08-22_11_08.npz", 1, "0808 22:11:08"),
    (ROOT / "model_tuning/data/extracted_bags/rosbag2_2026_08_10-21_45_06.npz", 1, "0810 21:45:06"),
    (ROOT / "model_tuning/data/extracted_bags/rosbag2_2026_08_10-21_45_57.npz", 1, "0810 21:45:57"),
    (ROOT / "model_tuning/data/extracted_bags/rosbag2_2026_08_10-21_52_23.npz", 1, "0810 21:52:23"),
]

OUTPUT = ROOT / "model_tuning/data/ifac_all_ackermann_bagdisjoint_train_test.npz"
def main():
 arrays=[];metadata=[];next_bag=0;columns=None;dt=None
 for path,split,label in SOURCES:
  z=np.load(path);a=z['samples'].copy();this_dt=float(z['dt'])
  if dt is None:dt=this_dt;columns=z['columns']
  if abs(this_dt-dt)>1e-9 or not np.array_equal(columns,z['columns']):raise RuntimeError('schema/dt mismatch')
  original=a[:,11].astype(int);mapping={old:next_bag+i for i,old in enumerate(np.unique(original))}
  a[:,11]=np.array([mapping[x] for x in original]);next_bag+=len(mapping);a[:,10]=split
  arrays.append(a);metadata.append({'path':str(path),'label':label,'split':'train' if split==0 else 'test','samples':len(a),'bag_ids':list(mapping.values())})
 out=np.concatenate(arrays);OUTPUT.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(OUTPUT,samples=out,dt=dt,columns=columns)
 train_samples=int((out[:,10]==0).sum());test_samples=int((out[:,10]==1).sum())
 report={'output':str(OUTPUT),'samples':len(out),'train_samples':train_samples,
         'test_samples':test_samples,'test_fraction':test_samples/len(out),
         'command_topic':'/ackermann_cmd','sources':metadata,
         'split_policy':'source-bag-disjoint according to SOURCES (0=train, 1=test)'}
 OUTPUT.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
