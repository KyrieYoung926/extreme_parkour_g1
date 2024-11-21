# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class H1RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 1024
        num_actions = 19
        n_priv_latent = 4 + 1 + 2*num_actions

        n_scan = 132
        n_priv = 3+3 +3
        n_proprio = 3 + 2 + 3 + 2 + 36 + 5
        history_len = 10

        # num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv #n_scan + n_proprio + n_priv #187 + 47 + 5 + 12 
        num_observations = 745

        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        obs_type = "og"


        
        
        
        history_encoding = True
        reorder_dofs = True
        
        
        # action_delay_range = [0, 5]

        # additional visual inputs 

        # action_delay_range = [0, 5]

        # additional visual inputs 
        include_foot_contacts = True
        
        randomize_start_pos = False
        randomize_start_vel = False
        randomize_start_yaw = False
        rand_yaw_range = 1.2
        randomize_start_y = False
        rand_y_range = 0.5
        randomize_start_pitch = False
        rand_pitch_range = 1.6

        contact_buf_len = 100

        next_goal_threshold = 0.2
        reach_goal_delay = 0.1
        num_future_goal_obs = 2

    class init_state:
        pos = [0.0, 0.0, 1.05] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'left_hip_yaw_joint': 0.0,
            'left_hip_roll_joint': 0.0,
            'left_hip_pitch_joint': -0.349,
            'left_knee_joint': 0.698,
            'left_ankle_joint': -0.349,
            'right_hip_yaw_joint': 0.0,
            'right_hip_roll_joint': 0.0,
            'right_hip_pitch_joint': -0.349,
            'right_knee_joint': 0.698,
            'right_ankle_joint': -0.349,
            'torso_joint': 0.0,
            'left_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.0,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 0.0,
            'right_shoulder_pitch_joint': 0.0,
            'right_shoulder_roll_joint': 0.0,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_joint': 0.0,
        }
        randomize_upperbody = False

        
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {
            'hip_yaw': 200,
            'hip_roll': 200,
            'hip_pitch': 200,
            'knee': 200,
            'ankle': 40,
            'torso': 300,
            'shoulder': 40,
            'elbow': 40,
        }
        damping = {
            'hip_yaw': 5,
            'hip_roll': 5,
            'hip_pitch': 5,
            'knee': 5,
            'ankle': 2,
            'torso': 6,
            'shoulder': 2,
            'elbow': 2,
        }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 20

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/urdf/h1.urdf'
        foot_name = "ankle"
        penalize_contacts_on = []
        terminate_after_contacts_on = ["pelvis", 
                                       "shoulder", 
                                    #    "elbow",
                                       "hip_roll", 
                                       "hip_pitch", 
                                       "knee",
                                       "torso",
                                       ]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        n_lower_body_dofs: int = 12

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 1
        only_positive_rewards = True
        tracking_sigma = 0.2 
        soft_dof_vel_limit = 1
        soft_torque_limit = 0.4
        max_contact_force = 200.0 
        min_dist = 0.1
        max_dist = 0.2

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_goal_vel = 2.0
            tracking_yaw = 0.5 
            orientation = -0.1 
            dof_acc = -2.5e-7

            
            lin_vel_z = -1.0 / 50
            ang_vel_xy = -0.05 / 50
            collision = -10.0 / 50
            action_rate = -0.1 / 50
            delta_torques = -1.0e-7 / 50
            torques = -0.00001 / 50
            hip_pos = -0.0  
            dof_error = -0.15
            dof_error_upper = -0.2

            feet_stumble = -1.0 / 50
            feet_edge = -1.0 / 50
            
            feet_distance = 0.2
            feet_contact_forces = -2e-3
            dof_pos_limits = -10
            dof_torque_limits = -0.1



class H1RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'h1_rough'
        experiment_name = 'rough_h1'
    class depth_encoder:
        if_depth = H1RoughCfg.depth.use_camera
        depth_shape = H1RoughCfg.depth.resized
        buffer_len = H1RoughCfg.depth.buffer_len
        hidden_dims = 512
        learning_rate = 1.e-3
        num_steps_per_env = H1RoughCfg.depth.update_interval * 24

    class estimator:
        train_with_estimated_states = True
        learning_rate = 1.e-4
        hidden_dims = [128, 64]
        priv_states_dim = H1RoughCfg.env.n_priv
        num_prop = H1RoughCfg.env.n_proprio
        num_scan = H1RoughCfg.env.n_scan

  
