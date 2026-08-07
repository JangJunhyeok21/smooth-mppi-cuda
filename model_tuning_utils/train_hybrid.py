#!/usr/bin/env python3
"""Fit SMPPI Pacejka parameters, then train an LSTM derivative residual."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter


NAMES = ("Cm0", "B_f", "C_f", "D_f", "E_f", "B_r", "C_r", "D_r", "E_r")


def load_dataset(path, args):
    z = np.load(path)
    a, dt = z["samples"].astype(np.float64), float(z["dt"])
    vx, vy, omega = a[:, 4], a[:, 5], a[:, 6]
    v = np.hypot(vx, vy)
    beta = np.arctan2(vy, np.maximum(vx, 1e-3))
    win = min(args.smooth_window, len(a) // 2 * 2 - 1)
    if win >= 5:
        v = savgol_filter(v, win, 3); beta = savgol_filter(beta, win, 3)
        omega = savgol_filter(omega, win, 3)
    state = np.column_stack((v, beta, omega))
    deriv = np.gradient(state, dt, axis=0)
    control = a[:, 7:9]
    valid = ((v >= args.min_speed) & (v <= args.max_data_speed) &
             (np.abs(omega) <= args.max_abs_omega) &
             (np.abs(deriv[:, 0]) <= args.max_abs_v_dot) &
             (np.abs(deriv[:, 1]) <= args.max_abs_beta_dot) &
             (np.abs(deriv[:, 2]) <= args.max_abs_omega_dot) &
             np.all(np.isfinite(np.column_stack((state, control, deriv))), axis=1))
    print(f"physical data filter retained {valid.sum()}/{len(valid)} samples")
    if valid.sum() < 100:
        raise SystemExit("fewer than 100 valid dynamic samples; inspect topics/frames and filter limits")
    return state[valid], control[valid], deriv[valid], dt


def classic_derivative(state, control, theta, fixed):
    v, beta, omega = state.T; steer, accel = control.T
    cm0, bf, cf, df, ef, br, cr, dr, er = theta
    mass, iz, lf, lr, g = fixed
    vx = np.maximum(v * np.cos(beta), 0.2)
    vy = v * np.sin(beta)
    af = steer - np.arctan2(vy + lf * omega, vx)
    ar = -np.arctan2(vy - lr * omega, vx)
    fzf, fzr = mass * g * lr / (lf + lr), mass * g * lf / (lf + lr)
    fyf = fzf * df * np.sin(cf * np.arctan(bf * af - ef * (bf * af - np.arctan(bf * af))))
    fyr = fzr * dr * np.sin(cr * np.arctan(br * ar - er * (br * ar - np.arctan(br * ar))))
    return np.column_stack((accel * (1.0 - cm0 * v),
                            (fyf * np.cos(steer) + fyr) / (mass * np.maximum(v, 0.5)) - omega,
                            (lf * fyf * np.cos(steer) - lr * fyr) / iz))


def fit_classic(state, control, observed, args):
    fixed = (args.mass, args.iz, args.lf, args.lr, 9.81)
    s, u, y = state, control, observed
    if len(s) > args.max_fit_samples:
        idx = np.linspace(0, len(s) - 1, args.max_fit_samples).astype(int)
        s, u, y = s[idx], u[idx], y[idx]
    scale = np.array([2.0, 4.0, 20.0])
    x0 = np.array([.04, 6., 1.4, .8, .1, 6., 1.4, .8, .1])
    lo = np.array([0., .1, .5, .1, -1., .1, .5, .1, -1.])
    hi = np.array([.5, 30., 3., 2., 1., 30., 3., 2., 1.])
    result = least_squares(lambda q: ((classic_derivative(s, u, q, fixed)-y)/scale).ravel(),
                           x0, bounds=(lo, hi), loss="soft_l1", max_nfev=args.max_nfev,
                           verbose=1)
    return result.x, fixed


def train_residual(state, control, observed, classic, args, output):
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit("PyTorch is required for residual training: pip install torch\n" + str(exc))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    features = np.column_stack((state, control, classic))
    targets = observed - classic
    cut = int(len(features) * (1.0 - args.val_fraction))
    xmean, xstd = features[:cut].mean(0), features[:cut].std(0).clip(1e-6)
    ymean, ystd = targets[:cut].mean(0), targets[:cut].std(0).clip(1e-6)
    x = ((features-xmean)/xstd).astype("float32")
    y = ((targets-ymean)/ystd).astype("float32")

    def sequences(begin, end):
        ids = np.arange(max(begin, args.seq_len-1), end)
        return np.stack([x[i-args.seq_len+1:i+1] for i in ids]), y[ids]
    xtr, ytr = sequences(0, cut); xva, yva = sequences(cut, len(x))
    cell = nn.LSTM
    class ResidualRNN(nn.Module):
        def __init__(self):
            super().__init__(); self.rnn = cell(x.shape[1], args.hidden, args.layers,
                                                batch_first=True, dropout=args.dropout if args.layers > 1 else 0.)
            self.head = nn.Linear(args.hidden, 3)
        def forward(self, z):
            return self.head(self.rnn(z)[0][:, -1])
    device = torch.device(args.device)
    model = ResidualRNN().to(device); opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.SmoothL1Loss(); best, patience = float("inf"), 0
    ds = torch.utils.data.TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr))
    loader = torch.utils.data.DataLoader(ds, args.batch_size, shuffle=True)
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); loss = loss_fn(model(xb.to(device)), yb.to(device)); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(torch.from_numpy(xva).to(device)), torch.from_numpy(yva).to(device)).item()
        print(f"epoch {epoch+1:03d}: val={val:.6f}")
        if val < best:
            best, patience = val, 0; torch.save(model.state_dict(), output / "residual_state.pt")
        else:
            patience += 1
            if patience >= args.patience: break
    model.load_state_dict(torch.load(output / "residual_state.pt", map_location=device, weights_only=True))
    model.eval().cpu()
    example = torch.zeros(1, args.seq_len, x.shape[1])
    torch.jit.trace(model, example).save(str(output / "residual_model.ts"))
    np.savez(output / "normalization.npz", input_mean=xmean, input_std=xstd,
             output_mean=ymean, output_std=ystd,
             input_names=np.array(["v", "beta", "omega", "steer", "accel",
                                   "classic_v_dot", "classic_beta_dot", "classic_omega_dot"]),
             output_names=np.array(["v_dot_residual", "beta_dot_residual", "omega_dot_residual"]))
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset"); p.add_argument("-o", "--output", default="model_tuning/output")
    p.add_argument("--rnn", choices=("lstm",), default="lstm",
                   help="recurrent residual architecture (LSTM only)")
    p.add_argument("--mass", type=float, default=3.74); p.add_argument("--iz", type=float, default=.04712)
    p.add_argument("--lf", type=float, default=.163); p.add_argument("--lr", type=float, default=.161)
    p.add_argument("--min-speed", type=float, default=.7); p.add_argument("--smooth-window", type=int, default=11)
    p.add_argument("--max-data-speed", type=float, default=12.0)
    p.add_argument("--max-abs-omega", type=float, default=12.0)
    p.add_argument("--max-abs-v-dot", type=float, default=15.0)
    p.add_argument("--max-abs-beta-dot", type=float, default=15.0)
    p.add_argument("--max-abs-omega-dot", type=float, default=80.0)
    p.add_argument("--max-fit-samples", type=int, default=30000); p.add_argument("--max-nfev", type=int, default=300)
    p.add_argument("--seq-len", type=int, default=20); p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2); p.add_argument("--dropout", type=float, default=.1)
    p.add_argument("--epochs", type=int, default=100); p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--val-fraction", type=float, default=.2); p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=7); p.add_argument("--classic-only", action="store_true")
    args = p.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    state, control, observed, dt = load_dataset(args.dataset, args)
    theta, fixed = fit_classic(state, control, observed, args)
    classic = classic_derivative(state, control, theta, fixed)
    result = {**dict(zip(NAMES, map(float, theta))), "mass": args.mass, "I_z": args.iz,
              "l_f": args.lf, "l_r": args.lr, "dt": dt,
              "classic_rmse": dict(zip(("v_dot", "beta_dot", "omega_dot"),
                                        np.sqrt(np.mean((observed-classic)**2, axis=0)).tolist()))}
    tire_lo = np.array([.1, .5, .1, -1., .1, .5, .1, -1.])
    tire_hi = np.array([30., 3., 2., 1., 30., 3., 2., 1.])
    if (np.any(np.isclose(theta[1:], tire_lo, rtol=0, atol=1e-3)) or
            np.any(np.isclose(theta[1:], tire_hi, rtol=0, atol=1e-3))):
        result["warning"] = ("one or more tire parameters reached a bound; do not deploy "
                             "before checking topic/frame/delay")
    (output / "classic_params.json").write_text(json.dumps(result, indent=2) + "\n")
    if not args.classic_only:
        result["normalized_validation_loss"] = train_residual(state, control, observed, classic,
                                                                args, output)
        (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
