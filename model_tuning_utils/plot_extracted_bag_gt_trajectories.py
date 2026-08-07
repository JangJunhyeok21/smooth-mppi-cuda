#!/usr/bin/env python3
"""Plot GT x/y trajectories for every extracted bag and the cleaned subset."""
import argparse, json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset"); p.add_argument("-o","--output",required=True)
    p.add_argument("--cleaning-report")
    a=p.parse_args(); z=np.load(a.dataset); raw=z["samples"]; bag=raw[:,11].astype(int)
    report=json.loads(Path(a.cleaning_report).read_text()) if a.cleaning_report else {"rejected_bags":{}}
    rejected={int(k) for k in report.get("rejected_bags",{})}
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    manifest_path=Path(a.dataset).with_suffix(".manifest.json")
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    bag_meta={int(x["bag_id"]):x for x in manifest.get("bags",[])}

    def grid(ids,filename,title,color):
        cols=4; rows=math.ceil(len(ids)/cols)
        fig,axes=plt.subplots(rows,cols,figsize=(4.2*cols,3.8*rows),squeeze=False)
        for ax,bid in zip(axes.flat,ids):
            ii=np.flatnonzero(bag==bid); xy=raw[ii,1:3]; split="test" if int(raw[ii[0],10]) else "train"
            ax.plot(xy[:,0],xy[:,1],lw=1.2,color=color)
            ax.scatter(xy[0,0],xy[0,1],s=22,c="#2ca02c",marker="o",label="start")
            ax.scatter(xy[-1,0],xy[-1,1],s=25,c="#d62728",marker="x",label="end")
            maxstep=np.linalg.norm(np.diff(xy,axis=0),axis=1).max(initial=0.)
            ax.set_title(f"bag {bid} | {split} | n={len(ii)}\nmax step={maxstep:.3f} m",fontsize=9)
            ax.set_aspect("equal",adjustable="datalim"); ax.grid(alpha=.25); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        for ax in axes.flat[len(ids):]: ax.axis("off")
        fig.suptitle(title,fontsize=15,y=.999); fig.tight_layout(); fig.savefig(out/filename,dpi=170,bbox_inches="tight"); plt.close(fig)

    all_ids=sorted(map(int,np.unique(bag))); clean_ids=[x for x in all_ids if x not in rejected]
    bad_ids=[x for x in all_ids if x in rejected]
    grid(clean_ids,"clean_bags_gt_trajectories.png","Clean extracted bags — GT trajectories","#1f77b4")
    if bad_ids: grid(bad_ids,"rejected_bags_gt_trajectories.png","Rejected pose-corrupted bags — raw GT","#d62728")

    fig,axes=plt.subplots(1,2,figsize=(15,7))
    for ax,split_value,label in zip(axes,(0,1),("Train","Test")):
        count=0
        for bid in clean_ids:
            ii=np.flatnonzero(bag==bid)
            if int(raw[ii[0],10])!=split_value: continue
            ax.plot(raw[ii,1],raw[ii,2],lw=.9,alpha=.7); count+=1
        ax.set_title(f"{label} GT overlay — {count} clean bags"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_aspect("equal",adjustable="datalim"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out/"clean_train_test_gt_overlay.png",dpi=180); plt.close(fig)
    summary={"dataset":a.dataset,"clean_bag_ids":clean_ids,"rejected_bag_ids":bad_ids,
             "clean_bags":len(clean_ids),"rejected_bags":len(bad_ids)}
    (out/"gt_trajectory_plot_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
