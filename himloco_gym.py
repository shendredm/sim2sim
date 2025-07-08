import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
#from humanoid import LEGGED_GYM_ROOT_DIR
#from humanoid.envs import XBotLCfg
import torch


class cmd:
    vx = 1.0
    vy = 0.0
    dyaw = 0.0

class Sim2simCfg():

    class sim_config:
        mujoco_model_path = f'/home/dhanu/Dhananjay/sim2sim/resources/unitree_a1/scene.xml'
        sim_duration = 60.0
        dt = 0.005
        decimation = 4

    class robot_config:
        kps = 40.0 * np.ones(12, dtype=np.double)
        kds = 1.0 * np.ones(12, dtype=np.double)
        tau_limit = 33.5 * np.ones(12, dtype=np.double)

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
        
        clip_observations = 100.0
        clip_actions = 100.0
    class env:
        num_actions = 12
        num_single_obs = 45
        num_history_observations = num_single_obs * 6

    class control:
        action_scale = 0.25

def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    reshape_actions_array = [3,4,5,0,1,2,9,10,11,6,7,8] #FR,FL,RR,RL
    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    hist_obs = torch.tensor(np.zeros((1, cfg.env.num_history_observations), dtype=np.float32))
    
    # Set the initial position to the home position
    default_pos = model.keyframe('home').qpos
    default_pos = default_pos[-cfg.env.num_actions:]  # Get the last 12 DOF positions
    
    count_lowlevel = 0

    for i in range(cfg.sim_config.decimation):
        start_q = data.qpos.astype(np.double)
        start_q = start_q[-cfg.env.num_actions:]  # Get the last 12 DOF positions
        start_dq = data.qvel.astype(np.double)
        start_dq = start_dq[-cfg.env.num_actions:]  # Get the last 12 DOF velocities

        target_q = default_pos.copy()
        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        tau = pd_control(target_q, start_q, cfg.robot_config.kps,
                         target_dq, start_dq, cfg.robot_config.kds)
        
        data.ctrl = tau
        mujoco.mj_step(model, data)
        viewer.render()
    #input()
    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]

        # 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:

            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi

            obs[0, 0] = cmd.vx * cfg.normalization.obs_scales.lin_vel
            obs[0, 1] = cmd.vy * cfg.normalization.obs_scales.lin_vel
            obs[0, 2] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
            obs[0, 3:6] = omega * cfg.normalization.obs_scales.ang_vel
            obs[0, 6:9] = gvec
            obs[0, 9:21] = q[reshape_actions_array] - default_pos[reshape_actions_array] * cfg.normalization.obs_scales.dof_pos
            obs[0, 21:33] = dq[reshape_actions_array] * cfg.normalization.obs_scales.dof_vel
            obs[0, 33:45] = action[reshape_actions_array]

            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
            obs = torch.tensor(obs, dtype=torch.float32)
            hist_obs = torch.cat((obs[:, :cfg.env.num_single_obs], hist_obs[:, :-cfg.env.num_single_obs]), dim=-1)

            action[:] = policy(torch.tensor(hist_obs))[0].detach().numpy()
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

            target_q = action[reshape_actions_array] * cfg.control.action_scale


        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        # Generate PD control
        tau = pd_control(target_q + default_pos, q, cfg.robot_config.kps,
                        target_dq, dq, cfg.robot_config.kds)  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()


if __name__ == '__main__':

    # Load the policy
    policy = torch.jit.load('/home/dhanu/Dhananjay/Gym/HIMLoco/legged_gym/logs/rough_a1/exported/policies/policy.pt')
    run_mujoco(policy, Sim2simCfg())