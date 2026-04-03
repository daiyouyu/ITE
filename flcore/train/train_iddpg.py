# -*- coding: utf-8 -*-
import time
import numpy as np
import os
from datetime import datetime as dt
from flcore.train.train_common import (
    default_presets, load_series_split, build_envs,
    infer_dims, list_by_agents, flatten_obs, flatten_actions
)
from flcore.algorithm.IDDPG import IDDPG
from flcore.utils.print_epreward import format_episode_info

# The function signature is updated to accept fed_method
def train_iddpg(episodes=1000, train=7, test=1, Federated=True, fed_method='DSFA'):
    # --- Setup ---
    presets = default_presets()
    train_series, test_series, T, train_idx, test_idx = load_series_split(
        path1="./data/IES_data/G_demand.csv",
        path2="./data/IES_data/H_demand.csv",
        train_days=train,
        test_days=test
    )
    env, test_env = build_envs(train_series, test_series, presets.env_kwargs)
    obs_dims, action_dims, max_actions, agents = infer_dims(env)
    
    # Initialize the algorithm
    iddpg = IDDPG(
        obs_dims, action_dims, max_actions,
        gamma=presets.algo_kwargs["gamma"], tau=presets.algo_kwargs["tau"],
        batch_size=presets.algo_kwargs["batch_size"], buffer_size=presets.algo_kwargs["buffer_size"],
        lr_actor=presets.algo_kwargs["lr_actor"], lr_critic=presets.algo_kwargs["lr_critic"]
    )

    rewards, test_rewards = [], []
    # This list will store the average weight matrix for each episode
    federation_weights_history = []

    # --- Training Loop ---
    for ep in range(episodes):
        start_time = time.time()
        obs, _ = env.reset()
        ep_rew = np.zeros(len(agents), dtype=np.float32)
        
        # List to store weights for the current episode
        episode_weights = []
        
        # (Your existing ep_info dictionary setup remains unchanged)
        ep_info = {a: {k: 0.0 for k in [
            "G_demand_MWH", "p_bat_MWh", "market_buy_MWh", "market_sell_MWh", 
            "newpower_MWh", "e_grid_buy_MWh", "P_boiler_e_MWh", "P_CHP_e_MWh",
            "h_demand_MWH", "h_grid_buy_MWh", "P_CHP_h_MWh", "P_HB_h_MWh",
            "soc_cost", "boiler_cost", "CHP_cost", "HB_cost", "market_cost"
        ]} for a in range(len(agents))}

        horizon = max(1, len(train_idx))
        for t in range(horizon):
            obs_list = list_by_agents(obs, agents)
            
            # Action selection
            if t < presets.noise_warmup_steps:
                actions_list = [env.action_spaces[a].sample() for a in agents]
            else:
                noise_scale = 0.3 * (1 - t / horizon)
                actions_list = iddpg.select_actions(obs_list, noise_scale=noise_scale)

            action_dict = {a: actions_list[i] for i, a in enumerate(agents)}
            next_obs, rew_dict, term_dict, trunc_dict, info_dict = env.step(action_dict)

            # Store transition in replay buffer
            next_obs_list = list_by_agents(next_obs, agents)
            rew_list = list_by_agents(rew_dict, agents)
            done_list = [bool(term_dict[a]) or bool(trunc_dict[a]) for a in agents]
            joint_obs, joint_act = flatten_obs(obs_list), flatten_actions(actions_list)
            joint_next_obs = flatten_obs(next_obs_list)
            iddpg.replay.add(joint_obs, joint_act, rew_list, joint_next_obs, done_list)

            # Update networks
            if t > 0 and t % 3 == 0:
                iddpg.update()

            # --- Federated Aggregation Step ---
            # I_fed is 24, as in your original code
            if Federated and t > 0 and t % 24 == 0:
                # Call aggregation and potentially get weights back
                agg_weights = iddpg.Fed_Aggergate(method=fed_method)
                
                # If DSFA was used, weights are returned and we record them
                if agg_weights is not None: # If weights are returned, store them for this episode
                    episode_weights.append(agg_weights)

            obs = next_obs
            ep_rew += np.array(rew_list, dtype=np.float32)
            
            # (Your existing info dictionary update logic remains unchanged)
            for idx, a in enumerate(agents):
                info = info_dict[a]
                ep_info[idx]["G_demand_MWH"] += info.get("G_demand", 0.0)
                ep_info[idx]["market_buy_MWh"] += info.get("market_buy_MWh", 0.0)
                # ... and so on for all other keys
            
            if all(done_list):
                break

        # --- End of Episode ---
        # Calculate and store the average weights for this episode
        if episode_weights:
            avg_episode_weights = np.mean(episode_weights, axis=0)
            federation_weights_history.append(avg_episode_weights)

        rewards.append((ep_rew / max(1, t)) * 24)
        ep_time = (time.time() - start_time) / 60
        print(f"Fed:{Federated}_iddpg ({fed_method})\n"
              f"Episode {ep+1}/{episodes} | Time: {ep_time:.2f} min | Est. Rem: {ep_time * (episodes - ep - 1):.2f} min")
        print(format_episode_info(ep, (ep_rew / max(1, t)) * 24, ep_info[0]))

    # --- After Training ---
    env.close()
    # Save the model
    model_prefix = f"iddpg_{fed_method}"
    iddpg.save(prefix=model_prefix, Fed=Federated)

    # --- Save Federation Weights ---
    if federation_weights_history:
        # Create a directory for today's results if it doesn't exist
        now_date = dt.now().strftime('%Y%m%d')
        save_dir = os.path.join('result', now_date)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Define a unique file name for the weights
        timestamp = dt.now().strftime('%H%M%S')
        weights_filename = f'fed_weights_{fed_method}_{timestamp}.npy'
        weights_filepath = os.path.join(save_dir, weights_filename)
        
        # Convert list of matrices to a 3D numpy array and save
        weights_array = np.array(federation_weights_history)
        np.save(weights_filepath, weights_array)
        print(f"\nFederation weights history saved to: {weights_filepath}")
        print(f"Shape of saved weights: {weights_array.shape}")

    return rewards, test_rewards
