# -*- coding: utf-8 -*-
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List

class BatteryEnvSingle(gym.Env):
    """
    单智能体电池+锅炉简化环境（不再持有全时序数据）
    - 每一步的外生量 L、R、P、sin_h、cos_h 由协调器在 step 前注入
    - 子环境只维护本智能体的内部状态：SOC、随机数、等
    """
    metadata = {"render_modes": []}

    def __init__(self,
                 dt_hours=1.0,
                 E_bat_MWh=3.0,
                 P_bat_max_MW=1.0,
                 eta_ch=0.95, eta_dis=1.05,
                 soc_min=0.1, soc_max=0.9, soc_init=0.1,
                 deg_cost_per_MW=7,
                 penalty_soc=0.0,
                 episode_len=24,          # 由协调器传全局长度
                 obs_norm=True,           # 仍保留，便于后续扩展
                 seed: int | None = None,
                 ):
        super().__init__()
        self.dt = float(dt_hours)
        self.E = float(E_bat_MWh)
        self.Pmax = float(P_bat_max_MW)
        self.P_es = 0.1
        self.eta_ch, self.eta_dis = float(eta_ch), float(eta_dis)
        self.soc_min, self.soc_max = float(soc_min), float(soc_max)
        self.soc_init = float(soc_init)
        self.deg_c = float(deg_cost_per_MW)
        self.penalty_soc = float(penalty_soc)
        self.obs_norm = bool(obs_norm)
        # 锅炉参数
        self.Fbmax = 2
        self.LHV = 5.8111

        # 时序（由协调器控制，这里只作回合步计数）
        self.episode_len = int(episode_len)
        self._t = 0
        self._soc = float(self.soc_init)

        # 随机性（每个 agent 独立 RNG）
        self._rng = np.random.default_rng(seed)
        self.cco2 = 0.0712
        # 外生量占位（由协调器在每步注入）
        self._exog = {"G_L": 0.0, "H_L": 0.0, "R": 0.0, "P": 0.0, "sin_h": 0.0, "cos_h": 0.0}
        # 三个尺度用于观测归一化（简化：交由协调器预归一化更合理；这里保留不强依赖）
        self._G_L_scale = 1.0
        self._H_L_scale = 1.0
        self._R_scale = 1.0
        self._P_scale = 1.0

        # 动作/观测空间
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0.0,0.0, 0.0, -10.0, self.soc_min, -1.0, -1.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([2.0,2.0, 2.0, 10000.0, self.soc_max, 1.0, 1.0, 1e6, 1.0], dtype=np.float32)        )

    # 由协调器调用，注入本步外生量和（可选）归一化尺度
    def set_exogenous(self, G_L,H_L, R, P, sin_h, cos_h, G_L_scale=1.0, H_L_scale=1.0,R_scale=1.0, P_scale=1.0):
        self._exog = {"G_L": float(G_L),
                      "H_L": float(H_L),
                      "R": float(R),
                      "P": float(P),
                      "sin_h": float(sin_h),
                      "cos_h": float(cos_h)}
        self._G_L_scale = max(1e-6, float(G_L_scale))
        self._H_L_scale = max(1e-6, float(H_L_scale))
        self._R_scale = max(1e-6, float(R_scale))
        self._P_scale = max(1e-6, float(P_scale))

    def _make_obs(self, market_price: float = 0.0, market_MWH: float = 0.0):
        G_L = self._exog["G_L"] / self._G_L_scale if self.obs_norm else self._exog["G_L"]
        H_L = self._exog["H_L"] / self._H_L_scale if self.obs_norm else self._exog["H_L"]
        R = self._exog["R"] / self._R_scale if self.obs_norm else self._exog["R"]
        P = self._exog["P"] / self._P_scale if self.obs_norm else self._exog["P"]
        return np.array([
            G_L,
            H_L,
            R,
            P,
            self._soc,
            self._exog["sin_h"],
            self._exog["cos_h"],
            float(market_price),
            float(market_MWH)
        ], dtype=np.float32)

    def reset(self, seed=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._soc = float(np.clip(self.soc_init, self.soc_min, self.soc_max))
        # 注意：协调器会在 reset 之后立刻 set_exogenous(t=0) 再读取观测
        return self._make_obs(0.0, 0.0), {}

    def step(self, action: np.ndarray):
        # 读取当步外生量（由协调器注入）
        G_L = self._exog["G_L"];
        H_L = self._exog["H_L"];
        R = self._exog["R"];
        P = self._exog["P"]

        # 1) 电池动作缩放
        a = float(np.clip(action[0], -1.0, 1.0))
        a = (a + 1) / 2               #调整到 0~1
        if self.P_es >= 0 and self._soc < self.soc_max:
            self.P_es = min(a * self.Pmax,(self.soc_max-self._soc) * self.E/self.eta_ch)
            eta = self.eta_ch
        else:
            self.P_es = max(a * -self.Pmax, (self.soc_min - self._soc) * self.E / self.eta_dis)
            eta = self.eta_dis
        soc =  (eta * self.P_es)/self.E
        self._soc = soc+self._soc

        # 锅炉动作

        boiler_a = float(np.clip(action[1], -1.0, 1.0))
        fuel_kgph, P_boiler_e = self._boiler_block(boiler_a)
        # CHP动作
        CHP_a = float(np.clip(action[2], -1.0, 1.0))
        P_CHP_e,P_CHP_h = self._CHP_block(CHP_a)
        # HB动作
        HB_a = float(np.clip(action[3], -1.0, 1.0))
        P_HB_h = self._HB_block(HB_a)

        #电力结算
        net_power = G_L - R + self.P_es - P_CHP_e - P_boiler_e   # MW（不裁零，留给市场清算层处理）
        #热力结算
        net_heat = H_L - P_CHP_h - P_HB_h

        # 本地成本（不含购售电；电费由协调器做市场结算）
        soc_cost  = self.deg_c * abs(self.P_es)           # 折旧
        ##锅炉
        boiler_cost  = self._boiler_cost(fuel_kgph,P_boiler_e)
        ##CHP成本
        CHP_cost = self._CHP_cost(P_CHP_e,P_CHP_h)
        ##HB成本
        HB_cost = self._HB_cost(P_HB_h)

        # 奖励仅先返回“本地项”，最终奖励由协调器：-(电力结算+本地项)
        local_cost = soc_cost + boiler_cost + CHP_cost + HB_cost    # 注意 local_pen 已是“奖励加项”，这里减回
        reward_placeholder = 0.0  # 真正 reward 由协调器重算
        #aboiler = boiler_cost / P_boiler_e
        #aCHP = CHP_cost / (P_CHP_h + P_CHP_e)
        #aHB = HB_cost / P_HB_h
        # 推进内部计步
        self._t += 1
        done = (self._t >= self.episode_len)

        # === 子环境决定：是否售电 / 售电报价（成本×1.1） / 购买外部电价 ===
        # 单位成本近似：以“用于供电的出力”作为分母做加权
        p_dis = max(0.0, self.P_es)  # 电池放电功率
        gen = max(0.0, P_boiler_e) + max(0.0 , P_CHP_e)  # 锅炉出力
        R_gen = max(0.0, R)
        R_cost = max(0.0, R_gen * 5)
        denom = max(1e-9, p_dis + gen + R_gen)

        a_sell = float(np.clip(action[4], -1.0, 1.0))
        a_buy  = float(np.clip(action[5], -1.0, 1.0))
        # 供需声明（MW）
        demand_MW = max(0.0, net_power)  # >0 需要购电
        offer_MW = max(0.0, -net_power) * 0.8  # <0 可对外售电
        # 当没有有效“可供电能”时，直接声明不卖，避免 unit_cost 除 0
        if offer_MW <= 1e-6 or denom <= 1e-3:
            ask_price = None
        else:
            # 单位成本：$/MWh，防爆夹紧（也可用 P 的区间）
            unit_cost = (soc_cost + boiler_cost + R_cost ) / max(1e-3, denom)
            markup = 1.0 + 0.5 * (a_sell + 1.0) / 2.0  # 1.0~1.5
            ask_price = float(np.clip(unit_cost * markup, 0.2 * P, 1.5 * P))

        if demand_MW <= 1e-6 :
            need_price = None# 外部电网价格（$/MWh），作为保底补足价
        else :
            markup = 0.8 * (a_buy + 1.0) / 2.0
            need_price = float(np.clip(P * markup, 0.0, P))

        obs_next = self._make_obs()
        info = {
            "G_demand": G_L,
            "P_boiler_e": P_boiler_e,
            "P_CHP_e": P_CHP_e,
            "p_bat": -self.P_es,
            "H_demand": H_L,
            "P_CHP_h": P_CHP_h,
            "P_HB_h": P_HB_h,
            "soc": self._soc,
            "newpower_gen": R,
            "grid_price": P,
            "net_power_MW": net_power,    # 给协调器做市场清算
            "net_heat":net_heat,
            "CHP_cost":CHP_cost,
            "HB_cost":HB_cost,
            "soc_cost": soc_cost,
            "boiler_cost": boiler_cost,
            "local_cost": local_cost,     # 便于协调器直接叠加
            # === 市场撮合所需 ===
            "offer_MW": offer_MW,                 # 可售电量（MW）
            "ask_price": ask_price,        # 售电报价（$/MWh）
            "need_price" : need_price,
            "demand_MW": demand_MW,               # 购电需求（MW）
        }
        return obs_next, float(reward_placeholder), bool(done), False, info

    def _boiler_block(self,  a_norm: float):
        u = (float(np.clip(a_norm, -1.0, 1.0)) + 1.0) / 2.0  # u∈[0,1]
        u = max(0.0, 2.0 * (u - 0.1))  # 去中心：u=0.5 -> 0，u=1->1，u=0->0
        M_bfw = u * self.Fbmax
        Hfuel = M_bfw * self.LHV
        noise = self._rng.integers(low=-2, high=2) * 0.01
        p_gen = Hfuel * (0.43 + noise)
        return M_bfw, p_gen

    def _boiler_cost(self,fuel_kgph,P_boiler_e):
        cf = 612
        gf = 0.8325
        boiler_cost = cf * fuel_kgph  + gf * self.cco2 * P_boiler_e
        return boiler_cost
    #定义热电联产模块
    def _CHP_block(self,  a_norm: float):
        #定义参数
        P_CHP_e_h = 8
        P_CHP_e_l = 0
        P_CHP_h_h = 3
        P_CHP_h_l = 0

        alpha_CHP = 1.2
        #获取动作
        u = (float(np.clip(a_norm, -1.0, 1.0)) + 1.0) / 2.0  # u∈[0,1]
        P_CHP_e = u * (P_CHP_e_h - P_CHP_e_l) + P_CHP_e_l
        P_CHP_h = alpha_CHP * P_CHP_e
        return P_CHP_e,P_CHP_h

    def _CHP_cost(self,CHP_e,CHP_h):
        a = 0.72
        b = 0.405
        c = 0.108
        d = 229.2
        e = 171.9
        f = 75
        k_CHP = 0.005
        CHP_cost = a*CHP_e*CHP_e + b*CHP_h*CHP_h +c*CHP_e*CHP_h + d*CHP_e + e*CHP_h + f
        cco2_cost = k_CHP * CHP_e * self.cco2
        return CHP_cost  + cco2_cost

    # 定义热泵模块
    def _HB_block(self, a_norm: float):
        # 定义参数
        P_HB_e_h = 15
        P_HB_e_l = 0

        # 获取动作
        u = (float(np.clip(a_norm, -1.0, 1.0)) + 1.0) / 2.0  # u∈[0,1]
        P_HB_h = u * (P_HB_e_h - P_HB_e_l) + P_HB_e_l
        return P_HB_h

    def _HB_cost(self,HB_h):
        a = 0.0171
        b = 230.5
        c = 75
        k_HB = 0.008
        HB_cost = a*HB_h*HB_h + b*HB_h +c
        cco2_cost = HB_h * k_HB * self.cco2
        return HB_cost + cco2_cost

    def render(self): pass

class MultiBatteryCoordinator(gym.Env):
    """
    多智能体协调器（统一发放 L/R/P/sin/cos）
    - reset(): 返回 {agent_id: obs}
    - step(action_dict): 收集各子环境的 net_power_MW、本地成本，统一做“电力结算”后再给奖励
    - 这里先按“外网电价直接结算”占位；你后续可替换为内部市场出清（式14/15）
    """
    def __init__(self, series: Dict[str, np.ndarray], n_agents: int, **single_kwargs):
        super().__init__()
        self.n_agents = int(n_agents)
        self.agents: List[str] = [f"agent_{i}" for i in range(self.n_agents)]

        # ===== 全局外生数据（同一份价格与负荷，可后续改为每园区不同 =====
        # series: {'L':..., 'R':..., 'P':..., 'sin_h':..., 'cos_h':...}
        self.series = series
        self.T = len(self.series[0]["P"])
        self.t = 0

        # 归一化尺度（统一由协调器计算并下发）
        self._G_L_scale = max(1.0, float(np.percentile(self.series[0]['G_L'], 99)))
        self._H_L_scale = max(1.0, float(np.percentile(self.series[0]['H_L'], 99)))
        self._R_scale = max(1.0, float(np.percentile(self.series[0]['R'], 99)))
        self._P_scale = max(1.0, float(np.percentile(self.series[0]['P'], 99)))

        # 创建 N 个独立子环境（不再传入 series）
        # episode_len 用全局 T 或你训练时的滑窗长度
        self.envs: Dict[str, BatteryEnvSingle] = {
            aid: BatteryEnvSingle(episode_len=self.T, **single_kwargs,
                                  seed=np.random.randint(1, 10_000_000))
            for aid in self.agents
        }

        # 动作/观测空间字典
        self.observation_spaces: Dict[str, spaces.Box] = {aid: env.observation_space for aid, env in self.envs.items()}
        self.action_spaces: Dict[str, spaces.Box] = {aid: env.action_space for aid, env in self.envs.items()}

    def _inject_exogenous_to_all(self, t: int):
        for i,(aid,env) in enumerate(self.envs.items()):
            G_L = float(self.series[i]["G_L"][t])
            H_L = float(self.series[i]['H_L'][t])
            R = float(self.series[i]["R"][t])
            P = float(self.series[i]["P"][t])
            sin_h = float(self.series[i]["sin_h"][t])
            cos_h = float(self.series[i]["cos_h"][t])
            env.set_exogenous(G_L,H_L, R, P, sin_h, cos_h,
                              G_L_scale=self._G_L_scale,
                              H_L_scale=self._H_L_scale,
                              R_scale=self._R_scale,
                              P_scale=self._P_scale)

    def reset(self, seed=None, options=None):
        self.t = 0
        obs = {}
        # 先 reset 子环境，再注入 t=0 的外生量并重取观测
        for aid, env in self.envs.items():
            env.reset(seed=(None if seed is None else (seed + hash(aid) % 9973)))
        self._inject_exogenous_to_all(self.t)
        for aid, env in self.envs.items():
            obs[aid] = env._make_obs(0.0, 0.0)
        return obs, {}

    def step(self, actions: Dict[str, np.ndarray]):
        # 注入当前时刻外生量
        self._inject_exogenous_to_all(self.t)

        # 先让每个子环境物理推进，得到“净功率”和本地成本
        obs_tmp, info_tmp, term_tmp, trunc_tmp = {}, {}, {}, {}
        for aid, env in self.envs.items():
            a = actions.get(aid, np.zeros(env.action_space.shape, dtype=np.float32))
            o, _, term, trunc, info = env.step(a)  # 奖励先忽略，由本层统一计算
            obs_tmp[aid] = o
            info_tmp[aid] = info
            term_tmp[aid] = term
            trunc_tmp[aid] = trunc

        # ==== 电力结算（协调器只做撮合；报价/是否出售由子环境info给出） ====
        dt_h = 1.0  # 与子环境步长一致；如有不同，可从任一env.dt读取

        # 1) 收集买卖与报价
        sellers = []  # (ask_price, aid, offer_MW)
        buyers = []  # (aid, demand_MW)
        tmp = {aid: {
            "buy_MWh": 0.0, "sell_MWh": 0.0, "cash_trade": 0.0,"check":0
        } for aid in self.agents}

        for aid in self.agents:

            inf = info_tmp[aid]
            if np.isnan(inf["grid_price"]):
                print(f"⚠️ NaN in price for {aid} at step {self.t}")
            offer = float(inf.get("offer_MW", 0.0))
            demand = float(inf.get("demand_MW", 0.0))
            ask = inf.get("ask_price", None)
            need = inf.get("need_price", None)

            if offer >= 1e-6 and ask is not None:
                sellers.append((float(ask), aid, offer))
            if demand >= 1e-6:
                buyers.append((float(need),aid, demand))

        # 2) 撮合（从最低报价卖家开始）
        sellers.sort(key=lambda x: x[0])  # 价格升序
        #sellers_offersum = sum(sellers[:][2])
        buyers.sort(key=lambda x: x[0] , reverse =True)
        #buyers_demandsum = sum(buyers[:][2])
        seller_left = {aid: offer for _, aid, offer in sellers}

        for (need_price,buyer_aid, need) in buyers:
            remaining = need
            for (ask, seller_aid, _) in sellers:
                if remaining <= 1e-9:
                    break
                cap = seller_left.get(seller_aid, 0.0)
                if cap <= 1e-9:
                    continue
                trade = min(remaining, cap)
                check = (ask+need_price)/2
                # 成交记录（MWh与现金）
                tmp[buyer_aid]["buy_MWh"] += trade * dt_h
                tmp[buyer_aid]["cash_trade"] += check * trade * dt_h  # 买家支出（正）
                tmp[seller_aid]["sell_MWh"] += trade * dt_h
                tmp[seller_aid]["cash_trade"] += check * trade * dt_h  # 卖家收入（正）

                # 更新余量/需求
                seller_left[seller_aid] = cap - trade
                remaining -= trade
            # 剩余由外网补足（见下节统一结算）

        # 3) 统一结算 & 奖励
        rews, infos = {}, {}
        for aid in self.agents:
            inf = info_tmp[aid]
            net_power = float(inf["net_power_MW"])
            net_heat = float(inf["net_heat"])
            local_cost = float(inf["local_cost"])
            p_grid_buy = float(inf.get("outside_price", inf.get("grid_price", 0.0)))

            # 内部成交统计
            market_buy_MWh = tmp[aid]["buy_MWh"]
            market_sell_MWh = tmp[aid]["sell_MWh"]
            market_cash = tmp[aid]["cash_trade"]  # 卖家为收入（正），买家为支出（正）


            # 外网补足（只有买家有）
            demand_MW = float(inf.get("demand_MW", max(0.0, net_power)))
            trade_buy_MW = market_buy_MWh / max(1e-9, dt_h)
            grid_buy_MW = max(0.0, demand_MW - trade_buy_MW)
            grid_cost = p_grid_buy * grid_buy_MW * dt_h

            # 卖不掉的富余电报废
            offer_MW = float(inf.get("offer_MW", max(0.0, -net_power)))
            trade_sell_MW = market_sell_MWh / max(1e-9, dt_h)
            surplus_MW = max(0.0, offer_MW - trade_sell_MW)
            # 废电惩罚
            surplus_cost = surplus_MW * p_grid_buy * 0.2 * dt_h
            # 费用口径：支出为正、收入为负
            # - 买家：elec_cost = grid_cost + market_cash（支出）
            # - 卖家：elec_cost = - market_cash（收入记负）
            #   统一写成：
            elec_cost = grid_cost - (market_cash if market_sell_MWh > 0 else 0.0) \
                        + (market_cash if market_buy_MWh > 0 else 0.0) + surplus_cost
            #外网补足热力
            heat_cost = max(0.0,net_heat) * 360
            total_cost = local_cost + elec_cost + heat_cost
            rew = - total_cost / 1000.0

            rews[aid] = float(rew)
            ii = dict(inf)
            ii.update({
                "elec_cost": elec_cost,
                "p_grid_buy": grid_cost,
                "h_grid_buy": max(0.0,net_heat),
                "market_buy_MWh": market_buy_MWh,
                "market_sell_MWh": market_sell_MWh,
                "market_cashflow": market_cash if market_sell_MWh > 0 else -market_cash,  # 卖家为 +收入；买家为 +支出
                "grid_buy_MWh": grid_buy_MW * dt_h,
                "surplus_dump_MWh": surplus_MW * dt_h,
                "total_cost": total_cost
            })
            infos[aid] = ii

        # 推进协调器时钟
        self.t += 1
        # 生成下一个观测（基于 t+1 的外生量）
        obs = {}
        if self.t < self.T:
            self._inject_exogenous_to_all(self.t)
            for aid, env in self.envs.items():
                market_buy = float(infos[aid]["market_buy_MWh"])
                market_sell = float(infos[aid]["market_sell_MWh"])
                market_cash = float(infos[aid]["market_cashflow"])

                avg_price = 0.0
                market_MWH = 0.0
                if market_buy > 1e-6:
                    avg_price  = -market_cash / max(1e-6, market_buy)
                    market_MWH = -market_buy
                elif market_sell > 1e-6:
                    avg_price  = +market_cash / max(1e-6, market_sell)
                    market_MWH = market_sell

                obs[aid] = env._make_obs(avg_price, market_MWH)

        else:
            obs = obs_tmp  # 已经到末尾，随便给占位即可

        return obs, rews, term_tmp, trunc_tmp, infos

