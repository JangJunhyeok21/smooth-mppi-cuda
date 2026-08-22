#!/usr/bin/env python3
"""Build the MPPI track contract without discarding explicit boundaries."""
import argparse
import csv
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--centerline",type=Path)
    parser.add_argument("--width-profile",type=Path)
    parser.add_argument(
        "--boundary-xy",type=Path,
        help="CSV with s,left_x,left_y,right_x,right_y. In this mode the "
             "explicit boundary coordinates are preserved exactly.")
    parser.add_argument("--centerline-output",type=Path)
    parser.add_argument("--width-profile-output",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.boundary_xy is not None:
        if args.centerline is not None or args.width_profile is not None:
            parser.error("--boundary-xy cannot be combined with --centerline/--width-profile")
        with args.boundary_xy.open(newline="") as stream:
            rows=list(csv.DictReader(stream))
        required={"left_x","left_y","right_x","right_y"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"Boundary CSV requires {sorted(required)}: {args.boundary_xy}")
        left=np.asarray([[float(row["left_x"]),float(row["left_y"])] for row in rows])
        right=np.asarray([[float(row["right_x"]),float(row["right_y"])] for row in rows])
        center=0.5*(left+right)
        half_width=0.5*np.linalg.norm(left-right,axis=1)
        width=np.c_[half_width,half_width]
    else:
        if args.centerline is None or args.width_profile is None:
            parser.error("provide --boundary-xy or both --centerline and --width-profile")
        center=np.loadtxt(args.centerline,delimiter=",",dtype=float)
        with args.width_profile.open(newline="") as stream:
            rows=list(csv.DictReader(stream))
        width=np.asarray([[float(row["left_width"]),float(row["right_width"])]
                          for row in rows])
        if center.shape!=(len(width),2):
            raise ValueError(f"centerline {center.shape} and width profile {width.shape} mismatch")
        tangent=np.roll(center,-1,axis=0)-np.roll(center,1,axis=0)
        tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-12)
        normal=np.c_[-tangent[:,1],tangent[:,0]]
        left=center+normal*width[:,0,None];right=center-normal*width[:,1,None]

    if args.centerline_output is not None:
        args.centerline_output.parent.mkdir(parents=True,exist_ok=True)
        np.savetxt(args.centerline_output,center,delimiter=",",fmt="%.8f")
    if args.width_profile_output is not None:
        args.width_profile_output.parent.mkdir(parents=True,exist_ok=True)
        segment=np.linalg.norm(np.diff(center,axis=0),axis=1)
        s=np.r_[0.0,np.cumsum(segment)]
        with args.width_profile_output.open("w",newline="") as stream:
            writer=csv.writer(stream);writer.writerow(("s","left_width","right_width"))
            for distance,w in zip(s,width):writer.writerow((f"{distance:.8f}",*w))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fields=("x_m","y_m","w_tr_left_m","w_tr_right_m","w_total_m",
            "left_x_m","left_y_m","right_x_m","right_y_m")
    with args.output.open("w",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(fields)
        for point,w,lpoint,rpoint in zip(center,width,left,right):
            writer.writerow((*point,*w,float(w.sum()),*lpoint,*rpoint))
    print(f"Saved {len(center)} MPPI track points to {args.output}")


if __name__=="__main__":main()
