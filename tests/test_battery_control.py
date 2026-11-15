"""储能动作、能量守恒和回合重置的回归测试。"""
import unittest

import numpy as np

from flcore.Env.multi_env import BatteryEnvSingle


class BatteryControlTests(unittest.TestCase):
    def test_charge_discharge_energy_and_limits(self) -> None:
        """不同时间步下均按效率守恒，SOC 边界限制实际功率。"""
        for dt in (0.25, 1.0, 2.0):
            with self.subTest(dt=dt):
                env = BatteryEnvSingle(dt_hours=dt, E_bat_MWh=5, P_bat_max_MW=3)
                env.reset()
                charged_energy = 0.0
                for _ in range(30):
                    previous_soc = env._soc
                    env._soc_block(-1.0)
                    charged_energy += env.P_es * dt
                    self.assertAlmostEqual((env._soc - previous_soc) * env.E, env.P_es * dt * env.eta_ch)
                    self.assertLessEqual(env.P_es, env.Pmax)
                self.assertAlmostEqual(env._soc, env.soc_max)
                self.assertAlmostEqual(env.P_es, 0.0)
                discharged_energy = 0.0
                for _ in range(30):
                    previous_soc = env._soc
                    env._soc_block(1.0)
                    discharged_energy -= env.P_es * dt
                    self.assertAlmostEqual((previous_soc - env._soc) * env.E, -env.P_es * dt / env.eta_dis)
                self.assertAlmostEqual(env._soc, env.soc_min)
                self.assertAlmostEqual(env.P_es, 0.0)
                self.assertAlmostEqual(discharged_energy, charged_energy * env.eta_ch * env.eta_dis)

    def test_idle_reset_disabled_and_invalid_action(self) -> None:
        env = BatteryEnvSingle(soc_init=0.5)
        env._soc_block(-0.5)
        env._soc_block(0.0)
        self.assertEqual(env.P_es, 0.0)
        env._soc_block(0.5)
        env.reset()
        self.assertEqual(env.P_es, 0.0)
        self.assertEqual(env._soc, 0.5)
        env.es_enabled = False
        env._soc_block(-1.0)
        self.assertEqual(env.P_es, 0.0)
        self.assertEqual(env._soc, 0.5)
        with self.assertRaises(ValueError):
            env._soc_block(float('nan'))

    def test_invalid_action_penalty(self) -> None:
        """仅处罚储能被 SOC 截断的能量，覆盖禁用、可行和部分执行动作。"""
        cases = [(0.1, 1.0, True, 1.5), (0.9, -1.0, True, 1.5),
                 (0.5, 0.1, True, 0.0), (0.5, 0.0, True, 0.0),
                 (0.1, 1.0, False, 0.0), (0.2, 1.0, True, 1.025)]
        for soc, action, enabled, expected_energy in cases:
            with self.subTest(soc=soc, action=action, enabled=enabled):
                env = BatteryEnvSingle(soc_init=soc, E_bat_MWh=5, P_bat_max_MW=3,
                                       dt_hours=0.5, es_enabled=enabled,
                                       battery_invalid_action_cost_per_MWh=100)
                actions = np.zeros(6, dtype=np.float32)
                actions[0] = action
                info = env.step(actions)[-1]
                self.assertAlmostEqual(info['battery_invalid_MWh'], expected_energy)
                self.assertAlmostEqual(info['battery_invalid_action_cost'], expected_energy * 100)

    def test_penalty_reaches_reward_only(self) -> None:
        """协调器奖励包含处罚，经济成本不因处罚系数改变。"""
        from flcore.train.train_common import default_presets, load_series_split, build_envs
        presets = default_presets()
        series, _, _, _, _ = load_series_split(
            path1='./data/IES_data/G_demand.csv', path2='./data/IES_data/H_demand.csv',
            train_days=1, test_days=0)
        env, other = build_envs(series, series, presets.env_kwargs)
        try:
            env.reset()
            actions = {aid: np.array([1, 0, 0, 0, 0, 0], dtype=np.float32) for aid in env.agents}
            _, rewards, _, _, infos = env.step(actions)
            for aid in env.agents:
                info = infos[aid]
                self.assertGreater(info['battery_invalid_action_cost'], 0)
                self.assertAlmostEqual(rewards[aid], -(info['total_cost'] + info['battery_invalid_action_cost']) / 1000)
            for single in env.envs.values():
                single.battery_invalid_action_cost_per_MWh = 0
            env.reset()
            _, unpenalized_rewards, _, _, unpenalized_infos = env.step(actions)
            for aid in env.agents:
                self.assertAlmostEqual(infos[aid]['total_cost'], unpenalized_infos[aid]['total_cost'])
                self.assertGreater(unpenalized_rewards[aid], rewards[aid])
        finally:
            env.close()
            other.close()

    def test_step_power_sign_and_demand(self) -> None:
        """充电增加外部购电需求，放电降低需求，且 info 符号与绘图一致。"""
        env = BatteryEnvSingle(soc_init=0.5, chp_enabled=False, boiler_E=0)
        env.set_exogenous(10, 0, 0, 100, 0, 1)
        for action, expected_sign in ((-0.1, -1), (0.1, 1)):
            env.reset()
            actions = np.zeros(6, dtype=np.float32)
            actions[0] = action
            _, _, _, _, info = env.step(actions)
            self.assertEqual(np.sign(info['p_bat']), expected_sign)
            self.assertAlmostEqual(info['net_power_MW'], 10 - info['p_bat'])


if __name__ == '__main__':
    unittest.main()
