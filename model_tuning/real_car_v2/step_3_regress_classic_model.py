#!/usr/bin/env python3
"""Step 3: regress and select the classic Pacejka baseline.

This script never uses reconstructed diagnostic CSV data.  It preserves the
bag-level train/validation/test split and selects one *global* eight-parameter
classic model by validation open-loop rollout error.
"""
from pathlib import Path
import json, os
import numpy as np
import torch
from scipy.optimize import differential_evolution, least_squares
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("DYNAMIC_SOURCE_DATA", ROOT / "model_tuning/data/dynamic_40ms_all_drive_source_20ms.npz"))
OUT = Path(os.environ.get("DYNAMIC_REGRESSION_OUT", ROOT / "model_tuning/results/dynamic_40ms_regression"))
SEED = 31
HORIZON = 25                    # 1.0 s at one 40 ms MPPI knot
# The archive now contains more than 200 discontinuity-safe segments. Keep the
# optimizer balanced per bag, but cap redundant windows so a full rerun remains
# practical after adding a day of data.
MAX_PER_BAG = 80
ADAM_RESTARTS = 3
ADAM_STEPS = 600
SURROGATE_SAMPLES = 400
SURROGATE_PROPOSALS = 40000

NAMES = ("B_f", "C_f", "D_f", "E_f", "B_r", "C_r", "D_r", "E_r")
BOUNDS = np.asarray((
    (.2, 30.), (.5, 2.5), (.05, 3.5), (-1., 1.),
    (.2, 30.), (.5, 2.5), (.05, 3.5), (-1., 1.)), dtype=np.float64)
REFERENCE = np.asarray((6., 1.3, 1.0, 0., 6., 1.3, 1.0, 0.))


def starts(data, split):
    features, bag, splits, valid = (data[k] for k in
                                    ("features", "bag_id", "split", "valid"))
    result = []
    for bag_id in np.unique(bag[splits == split]):
        candidate = np.asarray([
            index for index in range(len(features)-2*HORIZON)
            if bag[index] == bag_id and splits[index] == split
            and valid[index:index+2*HORIZON+1].all()
            and np.all(bag[index:index+2*HORIZON+1] == bag_id)
            and np.mean(np.abs(features[index:index+2*HORIZON, 0])) > .5], int)
        if len(candidate) > MAX_PER_BAG:
            candidate = candidate[np.linspace(
                0, len(candidate)-1, MAX_PER_BAG).astype(int)]
        result.extend(candidate[::3])
    return np.asarray(result, int)


def rollout_numpy(parameters, data, window_starts, config):
    feature = data["features"]
    state = feature[window_starts, :3].astype(np.float64).copy()
    applied_steer = feature[window_starts, 5].astype(np.float64).copy()
    speed_reference = state[:, 0].copy()
    prediction, ground_truth = [], []
    Bf, Cf, Df, Ef, Br, Cr, Dr, Er = parameters
    lf, lr, mass, iz = [float(config[key]) for key in
                        ("l_f", "l_r", "mass", "dynamic_mlp_I_z")]
    wheelbase = lf + lr
    front_load = mass*9.81*lr/wheelbase
    rear_load = mass*9.81*lf/wheelbase
    dt = .04
    for step in range(HORIZON):
        row = window_starts + 2*step
        command = feature[row, 3:5]
        steer_target = np.clip(
            float(config["kinematic_steer_scale"])*command[:, 0]
            + float(config["kinematic_steer_bias"]), -.55, .55)
        steer_rate = np.clip(
            (steer_target-applied_steer)/float(config["steer_servo_time_constant"]),
            -float(config["actuator_max_steer_rate"]),
            float(config["actuator_max_steer_rate"]))
        applied_steer = np.clip(applied_steer + steer_rate*dt, -.55, .55)
        speed_command = np.clip(command[:, 1], float(config["min_speed"]), 4.)
        tau = np.where(speed_command >= speed_reference,
                       float(config["speed_reference_accel_time_constant"]),
                       float(config["speed_reference_brake_time_constant"]))
        speed_reference += np.clip(
            (speed_command-speed_reference)/tau,
            -float(config["actuator_max_speed_reference_rate"]),
            float(config["actuator_max_speed_reference_rate"]))*dt
        vx, vy, yaw_rate = state.T
        ax = np.clip(float(config["speed_servo_kp"])
                     *(speed_reference-vx),
                     float(config["min_accel"]), float(config["max_accel"]))
        safe_vx = np.maximum(np.abs(vx), .5)
        alpha_front = applied_steer-np.arctan2(vy+lf*yaw_rate, safe_vx)
        alpha_rear = -np.arctan2(vy-lr*yaw_rate, safe_vx)
        front_term = Bf*alpha_front
        rear_term = Br*alpha_rear
        fy_front = front_load*Df*np.sin(Cf*np.arctan(
            front_term-Ef*(front_term-np.arctan(front_term))))
        fy_rear = rear_load*Dr*np.sin(Cr*np.arctan(
            rear_term-Er*(rear_term-np.arctan(rear_term))))
        ay = (fy_front*np.cos(applied_steer)+fy_rear)/mass
        yaw_accel = (lf*fy_front*np.cos(applied_steer)-lr*fy_rear)/iz
        state = np.column_stack((
            vx+(ax+vy*yaw_rate)*dt,
            vy+(ay-vx*yaw_rate)*dt,
            yaw_rate+yaw_accel*dt))
        prediction.append(state.copy())
        truth=feature[window_starts+2*(step+1), :3].copy()
        if "teacher_vy" in data.files:
            truth[:,1]=data["teacher_vy"][window_starts+2*(step+1)]
        ground_truth.append(truth)
    return np.stack(prediction, 1), np.stack(ground_truth, 1)


def relative_pose(states, scale):
    pose = np.zeros((len(states), states.shape[1], 3)); dt = .04
    for step in range(states.shape[1]):
        previous = pose[:, step-1] if step else np.zeros((len(states), 3))
        vx, vy, yaw_rate = states[:, step].T
        pose[:, step, 0] = previous[:, 0] + scale*(
            vx*np.cos(previous[:, 2])-vy*np.sin(previous[:, 2]))*dt
        pose[:, step, 1] = previous[:, 1] + scale*(
            vx*np.sin(previous[:, 2])+vy*np.cos(previous[:, 2]))*dt
        pose[:, step, 2] = previous[:, 2] + yaw_rate*dt
    return pose


def objective(parameters, data, window_starts, config, regularize=True):
    prediction, truth = rollout_numpy(parameters, data, window_starts, config)
    time_weight = np.linspace(.25, 1., HORIZON)[None, :, None]
    state_error = (prediction-truth)*np.asarray((.4, 2., 1.5))[None, None, :]
    huber = np.where(np.abs(state_error) < .3,
                     .5*state_error**2, .3*(np.abs(state_error)-.15))
    loss = float(np.mean(huber*time_weight))
    scale = float(config["kinematic_position_speed_scale"])
    position_error = (relative_pose(prediction, scale)[:, -1, :2]
                      - relative_pose(truth, scale)[:, -1, :2])
    loss += .8*float(np.mean(np.sum(position_error**2, axis=1)))
    if regularize:
        span = BOUNDS[:, 1]-BOUNDS[:, 0]
        loss += 2e-4*float(np.mean(((parameters-REFERENCE)/span)**2))
        # Keep front/rear small-slip gains in the same physical order without
        # forcing identical tires under unequal load/observability.
        gains = np.asarray((parameters[0]*parameters[1]*parameters[2],
                            parameters[4]*parameters[5]*parameters[6]))
        loss += 1e-4*float((np.log((gains[0]+1e-4)/(gains[1]+1e-4)))**2)
    return loss


def metrics(parameters, data, window_starts, config):
    prediction, truth = rollout_numpy(parameters, data, window_starts, config)
    state_error = np.abs(prediction[:, -1]-truth[:, -1])
    scale = float(config["kinematic_position_speed_scale"])
    position_error = np.linalg.norm(
        relative_pose(prediction, scale)[:, -1, :2]
        - relative_pose(truth, scale)[:, -1, :2], axis=1)
    return {"windows": len(window_starts),
            "state_mae": state_error.mean(0).tolist(),
            "state_p95": np.quantile(state_error, .95, axis=0).tolist(),
            "trajectory_mean_m": float(position_error.mean()),
            "trajectory_p95_m": float(np.quantile(position_error, .95))}


class Surrogate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, 128), torch.nn.SiLU(),
            torch.nn.Linear(128, 128), torch.nn.SiLU(),
            torch.nn.Linear(128, 1))
    def forward(self, value): return self.net(value).squeeze(-1)


def surrogate_search(data, train_starts, config, rng):
    subset = train_starts[np.linspace(
        0, len(train_starts)-1, min(240, len(train_starts))).astype(int)]
    unit = (np.arange(SURROGATE_SAMPLES)[:, None]
            + rng.random((SURROGATE_SAMPLES, 8)))/SURROGATE_SAMPLES
    for column in range(8): rng.shuffle(unit[:, column])
    samples = BOUNDS[:, 0]+unit*(BOUNDS[:, 1]-BOUNDS[:, 0])
    targets = np.asarray([objective(p, data, subset, config) for p in samples],
                         np.float32)
    x = torch.as_tensor(unit, dtype=torch.float32)
    target_mean, target_std = float(targets.mean()), max(float(targets.std()), 1e-6)
    y = torch.as_tensor((targets-target_mean)/target_std)
    torch.manual_seed(SEED); network = Surrogate(); optimizer = torch.optim.AdamW(
        network.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(1200):
        index = torch.randint(len(x), (min(256, len(x)),))
        loss = torch.nn.functional.smooth_l1_loss(network(x[index]), y[index])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    proposals = rng.random((SURROGATE_PROPOSALS, 8)).astype(np.float32)
    with torch.no_grad(): predicted = network(torch.from_numpy(proposals)).numpy()
    best = proposals[np.argpartition(predicted, 32)[:32]]
    physical = BOUNDS[:, 0]+best*(BOUNDS[:, 1]-BOUNDS[:, 0])
    return min(physical, key=lambda p: objective(p, data, train_starts, config))


def torch_rollout_loss(raw_parameters, data, window_starts, config, device):
    lower = torch.tensor(BOUNDS[:, 0], device=device, dtype=torch.float64)
    upper = torch.tensor(BOUNDS[:, 1], device=device, dtype=torch.float64)
    parameters = lower+(upper-lower)*torch.sigmoid(raw_parameters)
    feature = torch.as_tensor(data["features"], device=device, dtype=torch.float64)
    starts_tensor = torch.as_tensor(window_starts, device=device, dtype=torch.long)
    state = feature[starts_tensor, :3].clone(); applied = feature[starts_tensor, 5].clone()
    speed_reference = state[:, 0].clone(); predictions=[]; truths=[]
    Bf,Cf,Df,Ef,Br,Cr,Dr,Er=parameters
    lf,lr,mass,iz=[float(config[k]) for k in ("l_f","l_r","mass","dynamic_mlp_I_z")]
    front_load=mass*9.81*lr/(lf+lr); rear_load=mass*9.81*lf/(lf+lr); dt=.04
    for step in range(HORIZON):
        row=starts_tensor+2*step; command=feature[row,3:5]
        target=torch.clamp(float(config["kinematic_steer_scale"])*command[:,0]
                           +float(config["kinematic_steer_bias"]),-.55,.55)
        rate=torch.clamp((target-applied)/float(config["steer_servo_time_constant"]),
                         -float(config["actuator_max_steer_rate"]),
                         float(config["actuator_max_steer_rate"]))
        applied=torch.clamp(applied+rate*dt,-.55,.55)
        speed=torch.clamp(command[:,1],float(config["min_speed"]),4.)
        tau=torch.where(speed>=speed_reference,
            torch.full_like(speed,float(config["speed_reference_accel_time_constant"])),
            torch.full_like(speed,float(config["speed_reference_brake_time_constant"])))
        speed_reference=speed_reference+torch.clamp((speed-speed_reference)/tau,
            -float(config["actuator_max_speed_reference_rate"]),
            float(config["actuator_max_speed_reference_rate"]))*dt
        vx,vy,yaw_rate=state.unbind(1)
        ax=torch.clamp(float(config["speed_servo_kp"])*(speed_reference-vx),
                       float(config["min_accel"]),float(config["max_accel"]))
        safe=torch.clamp(torch.abs(vx),min=.5)
        af=applied-torch.atan2(vy+lf*yaw_rate,safe); ar=-torch.atan2(vy-lr*yaw_rate,safe)
        bf=Bf*af;br=Br*ar
        fyf=front_load*Df*torch.sin(Cf*torch.atan(bf-Ef*(bf-torch.atan(bf))))
        fyr=rear_load*Dr*torch.sin(Cr*torch.atan(br-Er*(br-torch.atan(br))))
        ay=(fyf*torch.cos(applied)+fyr)/mass
        yaw_accel=(lf*fyf*torch.cos(applied)-lr*fyr)/iz
        state=torch.stack((vx+(ax+vy*yaw_rate)*dt,
                           vy+(ay-vx*yaw_rate)*dt,yaw_rate+yaw_accel*dt),1)
        predictions.append(state)
        truth=feature[starts_tensor+2*(step+1),:3].clone()
        if "teacher_vy" in data.files:
            teacher_vy=torch.as_tensor(data["teacher_vy"],device=device,dtype=torch.float64)
            truth[:,1]=teacher_vy[starts_tensor+2*(step+1)]
        truths.append(truth)
    prediction=torch.stack(predictions,1);truth=torch.stack(truths,1)
    error=(prediction-truth)*torch.tensor((.4,2.,1.5),device=device)
    loss=torch.nn.functional.smooth_l1_loss(error,torch.zeros_like(error),beta=.3)
    # Match the black-box objective: recursively integrate body velocities so
    # Adam cannot improve vy/r while silently worsening the actual trajectory.
    scale=float(config["kinematic_position_speed_scale"])
    predicted_pose=torch.zeros((len(window_starts),3),device=device,dtype=torch.float64)
    truth_pose=torch.zeros_like(predicted_pose)
    for step in range(HORIZON):
        pvx,pvy,pr=prediction[:,step].unbind(1)
        tvx,tvy,tr=truth[:,step].unbind(1)
        predicted_pose=torch.stack((
            predicted_pose[:,0]+scale*(pvx*torch.cos(predicted_pose[:,2])-pvy*torch.sin(predicted_pose[:,2]))*.04,
            predicted_pose[:,1]+scale*(pvx*torch.sin(predicted_pose[:,2])+pvy*torch.cos(predicted_pose[:,2]))*.04,
            predicted_pose[:,2]+pr*.04),1)
        truth_pose=torch.stack((
            truth_pose[:,0]+scale*(tvx*torch.cos(truth_pose[:,2])-tvy*torch.sin(truth_pose[:,2]))*.04,
            truth_pose[:,1]+scale*(tvx*torch.sin(truth_pose[:,2])+tvy*torch.cos(truth_pose[:,2]))*.04,
            truth_pose[:,2]+tr*.04),1)
    loss=loss+.8*torch.mean(torch.sum(
        (predicted_pose[:,:2]-truth_pose[:,:2])**2,dim=1))
    # Tail-risk term: classic yaw error in a few hard corners dominates MPPI
    # open-loop heading even when mean state loss is small.
    endpoint_yaw_error=torch.abs(prediction[:,-1,2]-truth[:,-1,2])
    # A static classic parameter vector cannot selectively repair a few
    # bag-specific yaw outliers.  A large CVaR term made the complete test
    # distribution worse, so tail correction is left to the residual model.
    # The classic fit still receives dense state and integrated-position loss.
    YAW_ENDPOINT_CVAR_WEIGHT=0.0
    if YAW_ENDPOINT_CVAR_WEIGHT>0:
        tail_count=max(1,int(.10*len(endpoint_yaw_error)))
        loss=loss+YAW_ENDPOINT_CVAR_WEIGHT*torch.topk(endpoint_yaw_error,tail_count).values.mean()
    reference=torch.tensor(REFERENCE,device=device,dtype=torch.float64)
    loss=loss+2e-4*torch.mean(((parameters-reference)/(upper-lower))**2)
    return loss,parameters


def adam_search(data, train_starts, config):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subset=train_starts[np.linspace(0,len(train_starts)-1,
        min(700,len(train_starts))).astype(int)]
    best=None
    for restart in range(ADAM_RESTARTS):
        generator=torch.Generator().manual_seed(SEED+restart)
        raw=torch.nn.Parameter(torch.randn(
            8,generator=generator,dtype=torch.float64).to(device))
        optimizer=torch.optim.AdamW([raw],lr=.035,weight_decay=1e-5)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,ADAM_STEPS,eta_min=2e-4)
        for _ in range(ADAM_STEPS):
            loss,parameters=torch_rollout_loss(raw,data,subset,config,device)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([raw],5.);optimizer.step();scheduler.step()
        candidate=parameters.detach().cpu().numpy()
        score=objective(candidate,data,train_starts,config)
        if best is None or score<best[0]:best=(score,candidate)
    return best[1]


def validation_score(metric):
    return (metric["trajectory_mean_m"]+.5*metric["trajectory_p95_m"]
            +.2*metric["state_mae"][2]+.1*metric["state_p95"][2])


def main():
    OUT.mkdir(parents=True,exist_ok=True);data=np.load(DATA)
    config=yaml.safe_load((ROOT/"config/params.yaml").read_text())["/**"]["ros__parameters"]
    train,validation,test=(starts(data,index) for index in range(3));rng=np.random.default_rng(SEED)
    current=np.asarray((2.9844349007584565,1.3,.362611229414815,0.,
                        .3173165891873783,1.3,2.799999941680244,0.))
    # Existing robust optimizer retained strictly as a comparison baseline.
    de=differential_evolution(lambda p:objective(p,data,train,config),BOUNDS,
        seed=SEED,popsize=6,maxiter=35,tol=8e-4,polish=False,workers=1)
    ls=least_squares(lambda p:(rollout_numpy(p,data,train,config)[0]
        -rollout_numpy(p,data,train,config)[1]).ravel(),de.x,
        bounds=(BOUNDS[:,0],BOUNDS[:,1]),loss="soft_l1",f_scale=.3,max_nfev=100)
    candidates={"current":current,"de_robust_ls":ls.x,
                "adam_differentiable":adam_search(data,train,config),
                "mlp_surrogate":surrogate_search(data,train,config,rng)}
    comparison={}
    for name,parameters in candidates.items():
        comparison[name]={"parameters":dict(zip(NAMES,parameters.tolist())),
            "train":metrics(parameters,data,train,config),
            "validation":metrics(parameters,data,validation,config),
            "test":metrics(parameters,data,test,config)}
        comparison[name]["validation_score"]=validation_score(comparison[name]["validation"])
    winner=min(comparison,key=lambda name:comparison[name]["validation_score"])
    selected=np.asarray([comparison[winner]["parameters"][name] for name in NAMES])
    tolerance=.01*(BOUNDS[:,1]-BOUNDS[:,0])
    boundary={name:bool(abs(value-low)<=tol or abs(high-value)<=tol)
              for name,value,(low,high),tol in zip(NAMES,selected,BOUNDS,tolerance)}
    report={"model_dt":.04,"integration":"single Euler step at 0.04 s",
        "parameter_names":list(NAMES),"selection":"lowest held-out validation open-loop score",
        "selected_method":winner,"expanded_fitted":dict(zip(NAMES,selected.tolist())),
        "boundary_solution":boundary,"deployment_gate_passed":not any(boundary.values()),
        "fixed_parameters":{"mass":float(config["mass"]),"I_z":float(config["dynamic_mlp_I_z"]),
            "l_f":float(config["l_f"]),"l_r":float(config["l_r"])},
        "methods":comparison}
    (OUT/"advanced_params.json").write_text(json.dumps(report,indent=2)+"\n")
    # Canonical downstream dataset/deployer consumes params.json.
    (OUT/"params.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
