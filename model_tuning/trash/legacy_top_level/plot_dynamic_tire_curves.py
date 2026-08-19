#!/usr/bin/env python3
"""Plot the exact front/rear Pacejka curves used by dynamic_mlp_residual."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "model_tuning/results/ifac_all_drive_dynamic_regression_fixed_iz/slip_fy_curves.png"
USE_PLOT = True

MASS = 3.74
GRAVITY = 9.81
L_F = 0.163
L_R = 0.161

B_F, C_F, D_F, E_F = 2.046134538755946, 2.3269535456162314, 0.07780187058396759, 0.9999999841747856
B_R, C_R, D_R, E_R = 1.921957391698233, 1.730957097201443, 0.06617241635354548, -0.999999818200288


def pacejka(alpha, normal_load, b, c, d, e):
    ba = b * alpha
    return normal_load * d * np.sin(c * np.arctan(ba - e * (ba - np.arctan(ba))))


def main():
    import matplotlib.pyplot as plt

    alpha = np.linspace(-0.8, 0.8, 2001)
    alpha_deg = np.degrees(alpha)
    f_zf = MASS * GRAVITY * L_R / (L_F + L_R)
    f_zr = MASS * GRAVITY * L_F / (L_F + L_R)
    f_yf = pacejka(alpha, f_zf, B_F, C_F, D_F, E_F)
    f_yr = pacejka(alpha, f_zr, B_R, C_R, D_R, E_R)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for axis in axes[:, 0]:
        axis.plot(alpha_deg, f_yf, label=rf"Front: $F_{{zf}}={f_zf:.2f}$ N")
        axis.plot(alpha_deg, f_yr, label=rf"Rear: $F_{{zr}}={f_zr:.2f}$ N")
        axis.axhline(0, color="0.5", lw=.8);axis.axvline(0, color="0.5", lw=.8)
        axis.set_ylabel(r"$F_y$ [N]");axis.grid(alpha=.25);axis.legend()
    axes[0,0].set_xlim(-45,45);axes[0,0].set_title("Pacejka lateral force: full slip-angle range")
    axes[1,0].set_xlim(-15,15);axes[1,0].set_title("Pacejka lateral force: operating-range detail")

    axes[0,1].plot(alpha_deg,f_yf/f_zf,label=rf"Front $D_f={D_F:.4f}$")
    axes[0,1].plot(alpha_deg,f_yr/f_zr,label=rf"Rear $D_r={D_R:.4f}$")
    axes[0,1].axhline(0,color="0.5",lw=.8);axes[0,1].axvline(0,color="0.5",lw=.8)
    axes[0,1].set_xlim(-45,45);axes[0,1].set_ylabel(r"$F_y/F_z$");axes[0,1].set_title("Normalized lateral force")
    axes[0,1].grid(alpha=.25);axes[0,1].legend()

    stiffness_f=f_zf*B_F*C_F*D_F;stiffness_r=f_zr*B_R*C_R*D_R
    axes[1,1].plot(alpha_deg,f_yf,label="Front nonlinear")
    axes[1,1].plot(alpha_deg,stiffness_f*alpha,"--",label=rf"Front small-angle: {stiffness_f:.2f} N/rad")
    axes[1,1].plot(alpha_deg,f_yr,label="Rear nonlinear")
    axes[1,1].plot(alpha_deg,stiffness_r*alpha,"--",label=rf"Rear small-angle: {stiffness_r:.2f} N/rad")
    axes[1,1].set_xlim(-10,10);axes[1,1].set_ylim(-2,2);axes[1,1].set_ylabel(r"$F_y$ [N]")
    axes[1,1].set_title("Small-angle stiffness comparison");axes[1,1].grid(alpha=.25);axes[1,1].legend(fontsize=8)
    for axis in axes.flat:axis.set_xlabel(r"slip angle $\alpha$ [deg]")
    fig.suptitle("MPPI dynamic_mlp_residual Pacejka tire curves",y=.995)
    fig.subplots_adjust(left=.08,right=.97,bottom=.08,top=.93,hspace=.32,wspace=.25)
    OUTPUT_PATH.parent.mkdir(parents=True,exist_ok=True);fig.savefig(OUTPUT_PATH,dpi=200)
    print(f"saved: {OUTPUT_PATH}")
    print(f"F_zf={f_zf:.6f} N, F_zr={f_zr:.6f} N")
    print(f"front max |Fy|={np.max(np.abs(f_yf)):.6f} N, rear max |Fy|={np.max(np.abs(f_yr)):.6f} N")
    print(f"small-angle stiffness: front={stiffness_f:.6f} N/rad, rear={stiffness_r:.6f} N/rad")
    if USE_PLOT:plt.show()
    else:plt.close(fig)


if __name__ == "__main__":
    main()
