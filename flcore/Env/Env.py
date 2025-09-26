# -*- coding: utf-8 -*-
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import scipy.io
import math
from typing import Dict, Any


class EnergyEnv(gym.Env):
    """
    工业能源系统（确定性优化模型）→ Gymnasium 环境
    - 单步为“当前时刻 t 的静态优化”，episode 按时间序列推进（t=0..T-1）
    - 动作含离散与连续：风机台数、SHC 面积、TES 充放热功率、两台锅炉负荷、三档蒸汽外购
    - 奖励 = - (当小时总成本 + 碳税) - 大罚项 * 约束违背
    论文等式映射：锅炉(1)(2)，风机(8)，SHC(5)(6)，TES(7)，系统热平衡(9)~(15)，CAPEX(17)，
                 运行/碳排(18)(19)(20)，变量范围(16)
    """
    metadata = {"render.modes": []}

    def __init__(self, N: int = 1, data_dir: str = "./data"):
        super().__init__()
        # ==== 常数参数（对齐论文符号 / 你给的脚本）====
        # 焓(kJ/kg)
        self.Hss = 3407.34
        self.Hhs = 3164.02
        self.Hms = 2877.53
        self.Hls = 2742.5
        self.Hw = 104.93
        self.Hsc = 2400.0    # SC（凝汽器侧）近似口径
        self.Hbfw = 642.12

        # 锅炉效率回归 & 额定水量(kg/h)、燃料低位发热值(kJ/kg) —— 式(1)(2)
        self.ba = 0.0851
        self.bb = 0.0079
        self.Fbmax = 20000.0     # 给水最大流量（单台）
        self.LHV = 45200.0       # 燃料低位发热值

        # 经济&排放（单位与注释保持一致）
        self.cf = 0.21085               # $/kg fuel（我们按 kg/h × $/kg 计）
        self.chs = 0.0164               # $/kg HS 蒸汽
        self.cms = 0.01495              # $/kg MS 蒸汽
        self.cls = 0.01161              # $/kg LS 蒸汽
        self.cele = 0.0821              # $/kWh（电价，如暂不结算可忽略）
        self.gf = 3.2233                # kgCO2/kg fuel
        self.ghs = 0.1991               # kgCO2/kg HS
        self.gms = 0.1811               # kgCO2/kg MS
        self.gls = 0.1726               # kgCO2/kg LS
        self.gele = 0.4019              # kgCO2/kWh
        self.cco2 = 0.01                # $/kg CO2（碳税）

        # 风机功率曲线 —— 式(8)
        self.Vcin, self.Vrat, self.Vcout, self.Grat = 3.0, 12.0, 25.0, 1000.0  # kW per WT @ rated
        self.xwt_max = 100  # 台数上限

        # SHC 参数 —— 式(5)(6)
        self.ASHC_max = 50000.0  # m^2
        self.Fr = 0.5573
        self.effSHC = 0.84
        self.LoSHC = 4.6797
        self.Tfw, self.T0, self.xSE = 25.0, 25.0, 1.0
        self.cpw, self.rw = 4.1819, 2260.0
        self.Tsat, self.Tls, self.cps = 143.61, 145.52, 2.3175

        # TES：热能仓（以等效 kWh_th 度量） —— 式(7)
        rho = 1000.0
        V_TES = 100.0
        deltaT = 50.0
        self.QTES = rho * self.cpw * V_TES * deltaT / 3600.0  # kWh_th 容量
        self.effTES = 0.85
        # 将往返效率拆成充/放（对称）
        self.eta_c = math.sqrt(self.effTES)
        self.eta_d = math.sqrt(self.effTES)
        self.tes_max_rate = self.QTES  # kW_th，|rate|=1 小时充/放满

        # CAPEX 年金→小时均摊 —— 式(17)
        self.cwt = 1610.0  # $/kW
        self.cSHC = 78.0   # $/m^2
        self.cTES = 25.0   # $/kW
        self.fr = 0.03
        self.N = 10
        annuity = self.fr * (1 + self.fr) ** self.N / ((1 + self.fr) ** self.N - 1)
        # 若要把 CAPEX 作为“固定+与动作相关”的混合项，可以把台数/面积/功率从动作中取值计提
        capex_year = annuity * (self.xwt_max * self.Grat * self.cwt +
                                self.ASHC_max * self.cSHC +
                                self.tes_max_rate * self.cTES)
        self.capex_per_hour = capex_year / 8760.0  # $/h（默认先当固定项；需要也可改为按动作比例计）

        # ==== 数据加载 ====
        steam = scipy.io.loadmat(f"{data_dir}/steam_data.mat")
        self.steam_data = steam

        self.Fhs_gt201 = steam["Fhs_gt201"].flatten()
        self.Fsc_gt201 = steam["Fsc_gt201"].flatten()

        self.Fms_gt501 = steam["Fms_gt501"].flatten()
        self.Fsc_gt501= steam["Fsc_gt501"].flatten()
        self.Fms_gt501= steam["Fms_gt501"].flatten()

        self.Fhs_gt2201= steam["Fhs_gt2201"].flatten()
        self.Fsc_gt2201= steam["Fsc_gt2201"].flatten()

        self.Fms_gt2501= steam["Fms_gt2501"].flatten()
        self.Fsc_gt2501= steam["Fsc_gt2501"].flatten()

        self.Fms_gt601= steam["Fms_gt601"].flatten()
        self.Fsc_gt2601= steam["Fsc_gt2601"].flatten()

        self.Fhs_gt04= steam["Fhs_gt04"].flatten()
        self.Fhs_gt05= steam["Fhs_gt05"].flatten()
        self.Fhs_gt06= steam["Fhs_gt06"].flatten()
        self.Fhs_gt07= steam["Fhs_gt07"].flatten()

        self.Fms_gt08= steam["Fms_gt08"].flatten()
        self.Fms_gt09= steam["Fms_gt09"].flatten()
        self.Fms_gt10= steam["Fms_gt10"].flatten()
        self.Fms_gt11= steam["Fms_gt11"].flatten()

        self.M_WHRS_SS = steam["Fss_ba101"].flatten()
        self.M_WHRS_SS += steam["Fss_ba102"].flatten()
        self.M_WHRS_SS += steam["Fss_ba103"].flatten()
        self.M_WHRS_SS += steam["Fss_ba104"].flatten()
        self.M_WHRS_SS += steam["Fss_ba105"].flatten()
        self.M_WHRS_SS += steam["Fss_ba106"].flatten()
        self.M_WHRS_SS += steam["Fss_ba107"].flatten()
        self.M_WHRS_SS += steam["Fss_ba108"].flatten()
        self.M_WHRS_SS += steam["Fss_ba110"].flatten()
        self.M_WHRS_SS += steam["Fss_ba111"].flatten()
        self.M_WHRS_SS += steam["Fss_ba2101"].flatten()
        self.M_WHRS_SS += steam["Fss_ba2102"].flatten()
        self.M_WHRS_SS += steam["Fss_ba2103"].flatten()
        self.M_WHRS_SS += steam["Fss_ba2104"].flatten()

        self.EA_SS = steam["Fss_ea_other"].flatten()
        self.EA_HS = steam["Fhs_ea_other"].flatten()
        self.EA_MS = steam["Fms_ea128"].flatten() + steam["Fms_ea146"].flatten()+steam["Fms_da2145"].flatten() + steam["Fms_ea2170"].flatten()

        self.EA_LS = steam["FLS_ea212"].flatten()
        self.EA_LS += steam["FLS_ea234"].flatten()
        self.EA_LS += steam["FLS_ea413"].flatten()
        self.EA_LS += steam["FLS_ea2400"].flatten()
        self.EA_LS += steam["FLS_ea2460"].flatten()
        self.EA_LS += steam["FLS_ea2170"].flatten()
        self.EA_LS += steam["FLS_ea2720"].flatten()
        self.EA_LS += steam["FLS_ea2131"].flatten()
        self.EA_LS += steam["FLS_ea2276"].flatten()
        self.EA_LS += steam["FLS_ea450"].flatten()
        self.EA_LS += steam["FLS_ea2440"].flatten()



        # 风/辐照
        inte = scipy.io.loadmat(f"{data_dir}/inte.mat")
        self.inte_wind = np.array(inte["inte_wind"]).flatten()
        self.inte_radi = np.array(inte["inte_radi"]).flatten()
        # 典型日：取 idx=57 的24小时（对齐你MATLAB逻辑：前一天18点到当天18点）
        # 若没有idx概念，则取任意连续24点

        self.T = 24*365
        # 用于归一化的尺度
        self.wind_max = max(1.0, float(np.max(self.inte_wind)))
        self.radi_max = max(1.0, float(np.max(self.inte_radi)))

        # ==== Gym 空间 ====
        # 动作：Dict
        #  - xwt: 风机台数（离散 0..xwt_max）
        #  - ashc: SHC面积 [0, ASHC_max]
        #  - tes_rate: TES充放热功率 [-tes_max_rate, +tes_max_rate]，kW_th，正为充，负为放
        #  - bfw: 两台锅炉给水 [0, Fbmax] kg/h
        #  - vl: 三档蒸汽阀门开关比例
        self.action_space = spaces.Dict({
            "xwt": spaces.Discrete(self.xwt_max + 1),
            "ashc": spaces.Box(low=np.array([0.0], dtype=np.float32),
                               high=np.array([self.ASHC_max], dtype=np.float32),
                               dtype=np.float32),
            "tes_rate": spaces.Box(low=np.array([-self.tes_max_rate], dtype=np.float32),
                                   high=np.array([ self.tes_max_rate], dtype=np.float32),
                                   dtype=np.float32),
            "bfw": spaces.Box(low=np.array([0.0, 0.0], dtype=np.float32),
                              high=np.array([self.Fbmax, self.Fbmax], dtype=np.float32),
                              dtype=np.float32),
            "vl": spaces.Box(low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                             high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                             dtype=np.float32)  # 上限可按现场改
        })

        # 观测：Box
        # [t_norm, wind_norm, radi_norm, soc_frac, demand_norm, hs_im_nom, ms_im_nom, ls_im_nom]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0,   0.0,   0.0,   0.0], dtype=np.float32),
            high=np.array([1.0, 1.2, 1.2, 1.0, 5.0,  5e4,   5e4,   5e4], dtype=np.float32),
            dtype=np.float32
        )

        # 内部状态
        self.t = 0
        self.soc = 0.5 * self.QTES  # kWh_th
        self._last_obs = None

        self.reset()

    # ---- Reset ----
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.soc = 0.5 * self.QTES
        obs = self._obs()
        self._last_obs = obs
        return obs, {}

    # ---- Step ----
    def step(self, action: Dict[str, Any]):
        # 读取动作并裁剪
        xwt = int(np.clip(int(action["xwt"]), 0, self.xwt_max))
        ashc = float(np.clip(float(action["ashc"]), 0.0, self.ASHC_max))
        tes_rate = float(np.clip(float(action["tes_rate"]), -self.tes_max_rate, self.tes_max_rate))
        bfw = np.asarray(action["bfw"], dtype=float)
        bfw = np.clip(bfw, 0.0, self.Fbmax)
        vl = np.asarray(action["vl"], dtype=float)  # [hs, ms, ls] kg/h
        vl = np.clip(vl, 0.0, np.array([1, 1, 1], dtype=float))

        # 当前时刻外部输入
        wind = float(self.inte_wind[self.t])
        radi = float(self.inte_radi[self.t])


        # 锅炉（两台）
        fuel1_kgph, Mbfss_1, nb1 = self._boiler_block(bfw[0])
        fuel2_kgph, Mbfss_2, nb2 = self._boiler_block(bfw[1])
        fuel_kgph = fuel1_kgph + fuel2_kgph
        Mbfss = Mbfss_1 + Mbfss_2

        # 风机（信息用途；如需要可用于电侧结算）
        gwt_single_kW = self._wind_power(wind)
        gwt_total_kW = xwt * gwt_single_kW

        # 太阳能热产生低压蒸汽
        M_SHC_LS = self._solar_heat_kW(radi, ashc)

        # TES 充放热 & SOC #TODO
        p_dis, p_chg, soc_next = self._tes_block(tes_rate)

        # 加载负荷

        # 蒸汽降级
        hs_vl, ms_vl, ls_vl = vl.tolist()
        SS_out = -(self.M_WHRS_SS[self.t] + Mbfss - self._load_SS() - self.EA_SS[self.t])
        # 各种平衡
        e_hs = self._letdown_volve(self.Hss,self.Hhs,hs_vl * (-SS_out))

        hs_im = -(e_hs - self._load_HS() - self.EA_HS[self.t])
        e_ms = self._letdown_volve(self.Hhs,self.Hms,ms_vl * (-hs_im) if hs_im < 0 else 0)

        ms_im = -(e_ms - self._load_MS() - self.EA_MS[self.t])
        e_ls = self._letdown_volve(self.Hms,self.Hls,ls_vl * ms_im)

        ls_im =-(M_SHC_LS + e_ls - self._load_LS() - self.EA_LS[self.t])
        G_im = (-self._load_ls_kW() + gwt_total_kW)

        # 运营成本（$/h）
        ope = (
                self.cf * fuel_kgph
                + self.chs * max(0, hs_im)
                + self.cms * max(0, ms_im)
                + self.cls * max(0, ls_im)
                + self.cele * max(0, G_im)
            # 可选：电力侧（若定义了 Elec_demand，可加入 cele*Elec_purchase）
        )
        # 排放（kgCO2/h）
        emiss = (
                self.gf * fuel_kgph
                + self.ghs * max(0, hs_im)
                + self.gms * max(0, ms_im)
                + self.gls * max(0, ls_im)
                + self.gele * max(0, G_im)
        )

        Cost = ope + self.cco2 * emiss

        total_cost = Cost

        reward = -float(total_cost)

        # 推进时间与 SOC
        self.soc = soc_next
        self.t += 1
        terminated = bool(self.t >= self.T)
        truncated = False

        info = {
            "t": self.t-1,

        }

        obs = self._terminal_obs() if terminated else self._obs()
        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    # ---- 观测 ----
    def _obs(self):
        t_norm = self.t / (self.T - 1)
        wind = float(self.inte_wind[self.t])
        radi = float(self.inte_radi[self.t])
        wind_norm = wind / self.wind_max
        radi_norm = radi / self.radi_max
        soc_frac = self.soc / max(1e-6, self.QTES)
        raw_demand = self._load_ls_kW(self.t)
        demand_norm = float(np.clip(raw_demand / (self._demand_scale + 1e-9), 0.0, 5.0))

        # 名义进口（如果只有标量，就复制）
        hs_nom = float(self.Fhs_im_arr[self.t % len(self.Fhs_im_arr)])
        ms_nom = float(self.Fms_im_arr[self.t % len(self.Fms_im_arr)])
        ls_nom = float(self.Fls_im_arr[self.t % len(self.Fls_im_arr)])

        obs = np.array(
            [t_norm, wind_norm, radi_norm, soc_frac, demand_norm, hs_nom, ms_nom, ls_nom],
            dtype=np.float32
        )
        return obs

    def _terminal_obs(self):
        return self._last_obs if self._last_obs is not None else self._obs()

    # ================== 单元函数 ==================
    # 锅炉 —— 式(1)(2)
    def _boiler_block(self, M_bfw: float):
        """
        输入：给水流量 M_bfw (kg/h)
        返回：fuel_kgph, heat_kW, nb
        fuel_kgph = M_bfw*(Hss - Hbfw) / (LHV * nb)
        heat_kW   = M_bfw*(Hss - Hbfw) / 3600
        """
        M_bfw = float(np.clip(M_bfw, 0.0, self.Fbmax))
        r = M_bfw / self.Fbmax
        nb = r / ((1.0 + self.bb) * r + self.ba)  # 效率回归
        nb = float(np.clip(nb, 1e-6, 0.98))       # 安全裁剪
        fuel_kgph = M_bfw * (self.Hss - self.Hbfw) / (self.LHV * nb)   # kg/h
        #炉子产生高压蒸汽
        Mbfss = fuel_kgph * (self.LHV * nb) / (self.Hss - self.Hbfw)               # kWh/h
        return fuel_kgph, Mbfss, nb

    # 风机 —— 式(8)
    def _wind_power(self, V: float) -> float:
        """单台风机功率（kW），三段式（立方段在 Vcin~Vrat）"""
        if V < self.Vcin or V >= self.Vcout:
            return 0.0
        if V < self.Vrat:
            den = (self.Vrat**3 - self.Vcin**3)
            return self.Grat * max(0.0, (V**3 - self.Vcin**3) / (den + 1e-9))
        return self.Grat

    # SHC —— 式(5)(6)
    def _solar_heat_kW(self, Ra: float, ASHC: float) -> float:
        """
        QSHC ≈ ASHC * Fr * (effSHC*Ra - LoSHC*(Tfw-T0)*xSE) * 3.6   (kWh/h)
        注：Ra 单位 W/m^2。为与原始 MATLAB 一致，这里沿用 *3.6 的经验系数。
        """
        QSHC = ASHC * self.Fr * (self.effSHC * Ra - self.LoSHC * (self.Tfw - self.T0) * self.xSE) * 3.6
        #用太阳能产生LS
        FSHC = QSHC / ((self.cpw * (self.Tsat - self.Tfw) + self.cps * (self.Tls - self.Tsat) + self.rw))
        return float(max(0.0, FSHC))

    # TES —— 式(7)
    def _tes_block(self, tes_rate_kW: float):
        """
        输入：tes_rate_kW，>0 充，<0 放
        返回：(P_dis, P_chg, soc_next)
        """
        rate = float(np.clip(tes_rate_kW, -self.tes_max_rate, self.tes_max_rate))
        soc = self.soc

        if rate < 0:  # 放热
            need_dis = -rate
            can_dis_by_soc = soc * self.eta_d  # 1小时时间窗内，最多可按eta_d放出
            p_dis = min(need_dis, can_dis_by_soc, self.tes_max_rate)
            # 放出 p_dis，对应 SOC 减少 p_dis/eta_d
            soc_next = soc - p_dis / max(1e-9, self.eta_d)
            return p_dis, 0.0, float(np.clip(soc_next, 0.0, self.QTES))
        else:         # 充热
            need_chg = rate
            can_chg_by_cap = (self.QTES - soc) / max(1e-9, self.eta_c)
            p_chg = min(need_chg, can_chg_by_cap, self.tes_max_rate)
            # 充入 p_chg，SOC 增加 eta_c*p_chg
            soc_next = soc + self.eta_c * p_chg
            return 0.0, p_chg, float(np.clip(soc_next, 0.0, self.QTES))

    # 代表性“动力负荷”（可换成你的实际 Guser）—— 用你给的 MATLAB 近似式
    def _load_ls_kW(self, t: int) -> float:
        """
        目前返回常量 base_demand_kW（kWh/h）。self.
        你可以改为：逐时序列（按 steam_data 的逐时蒸汽进/出计算），或外部传入。
        """
        Guser = [0]*14
        Guser[1] = ((self.Fhs_gt201[self.t] + self.Fsc_gt201[self.t]) * self.Hss - self.Fhs_gt201[self.t] * self.Hhs - self.Fsc_gt201[self.t] * self.Hsc) / 3600
        Guser[2] = ((self.Fms_gt501[self.t] + self.Fsc_gt501[self.t]) * self.Hss - self.Fms_gt501[self.t] * self.Hms - self.Fsc_gt501[self.t] * self.Hsc) / 3600
        Guser[3] = ((self.Fhs_gt2201[self.t] + self.Fsc_gt2201[self.t]) * self.Hss - self.Fhs_gt2201[self.t] * self.Hhs - self.Fsc_gt2201[self.t] * self.Hsc) / 3600

        Guser[4] = ((self.Fms_gt2501[self.t] + self.Fsc_gt2501[self.t]) * self.Hhs - self.Fms_gt2501[self.t] * self.Hms - self.Fsc_gt2501[self.t] * self.Hsc) / 3600
        Guser[5] = self.Fms_gt601[self.t] * (self.Hhs - self.Hms) / 3600
        Guser[6] = self.Fsc_gt2601[self.t] * (self.Hhs - self.Hsc) / 3600
        Guser[7] = (self.Hhs - self.Hls) * self.Fhs_gt04[self.t] / 3600
        Guser[8] = (self.Hhs - self.Hls) * self.Fhs_gt05[self.t] / 3600
        Guser[9] = (self.Hhs - self.Hls) * self.Fhs_gt06[self.t] / 3600
        Guser[10] = (self.Hhs - self.Hls) * self.Fhs_gt07[self.t] / 3600

        Guser[11] = (self.Hms - self.Hls) * self.Fms_gt08[self.t] / 3600
        Guser[12] = (self.Hms - self.Hls) * self.Fms_gt09[self.t] / 3600
        Guser[13] = (self.Hms - self.Hls) * self.Fms_gt10[self.t] / 3600
        Guser[0] = (self.Hms - self.Hls) * self.Fms_gt11[self.t] / 3600

        return max(0.0, sum(Guser))

    def _load_SS(self):
        ST_1 = self.Fhs_gt201[self.t] + self.Fsc_gt201[self.t]
        ST_2 = self.Fms_gt501[self.t] + self.Fsc_gt501[self.t]
        ST_3 = self.Fhs_gt2201[self.t] + self.Fsc_gt2201[self.t]
        return ST_1 + ST_2 + ST_3

    def _load_HS(self):

        ST_1 = self.Fhs_gt201[self.t]
        ST_3 = self.Fms_gt501[self.t]
        ST_4 = self.Fms_gt2501[self.t] + self.Fsc_gt2501[self.t]
        ST_5 = self.Fms_gt601[self.t]
        ST_6 = self.Fsc_gt2601[self.t]
        ST_7 = self.Fhs_gt04[self.t]
        ST_8 = self.Fhs_gt05[self.t]
        ST_9 = self.Fhs_gt06[self.t]
        ST_10 = self.Fhs_gt07[self.t]
        return  ST_10 + ST_9 + ST_8 + ST_7 + ST_6 + ST_5 + ST_4 - ST_3 - ST_1

    def _load_MS(self):
        ST_2 = self.Fms_gt501[self.t]
        ST_4 = self.Fms_gt2501[self.t]
        ST_5 = self.Fms_gt601[self.t]
        ST_11 =  self.Fms_gt08[self.t]
        ST_12 =  self.Fms_gt09[self.t]
        ST_13 =  self.Fms_gt10[self.t]
        ST_14 =  self.Fms_gt11[self.t]

        return ST_11 + ST_12 + ST_13 + ST_14 - ST_4 -ST_5 - ST_2

    # =============== 辅助函数 ===============
    def _letdown_volve(self,H_in,H_out,M_lv_in):
        M_lv_out = ((H_in-self.Hw) / (H_out - self.Hw )) * M_lv_in
        return M_lv_out

    def _safe_vec(self, mat: dict, key: str, default=None) -> np.ndarray:
        if default is None:
            default = []
        if key in mat:
            arr = np.array(mat[key]).flatten().astype(float)
            if arr.size == 0:
                return np.array(default, dtype=float)
            return arr
        return np.array(default, dtype=float)

    def _estimate_base_demand_from_steamdata(self, steam: dict) -> float:
        """
        尝试复现你 MATLAB 里 Guser(1..14) 的“第一行”合计热负荷（kWh/h）。
        如果缺字段，则回退为 5000 kWh/h。
        """
        try:
            # 简化：取若干关键点（若缺失则跳过），用你公式口径估算
            # 这里仅演示两三项，完整 14 项请按你的字段补足
            get = lambda k: float(np.array(steam[k]).flatten()[0]) if k in steam else None
            Fhs_gt201 = get("Fhs_gt201")
            Fsc_gt201 = get("Fsc_gt201")
            Fms_gt501 = get("Fms_gt501")
            Fsc_gt501 = get("Fsc_gt501")

            parts = []
            if Fhs_gt201 is not None and Fsc_gt201 is not None:
                g1 = ((Fhs_gt201 + Fsc_gt201) * self.Hss - Fhs_gt201 * self.Hhs - Fsc_gt201 * self.Hsc) / 3600.0
                parts.append(g1)
            if Fms_gt501 is not None and Fsc_gt501 is not None:
                g2 = ((Fms_gt501 + Fsc_gt501) * self.Hss - Fms_gt501 * self.Hms - Fsc_gt501 * self.Hsc) / 3600.0
                parts.append(g2)

            if len(parts) == 0:
                return 5000.0
            val = float(np.sum(parts))
            # 防止过小
            return max(2000.0, val)
        except Exception:
            return 5000.0

class BatteryArbEnv(gym.Env):
    """
    状态 s_t = [ L_t, R_t, P_t, SOC_t, sin(h), cos(h) ]
    动作 a_t = 归一化实数 ∈ [-1, 1]，映射到 p_bat_t ∈ [-Pmax, +Pmax] (MW)
    动力学:
        p_grid_t = L_t - R_t - p_bat_t
        SOC_{t+1} = SOC_t + (charge_or_discharge_energy / E_bat)
    奖励:
        r_t = -( P_t * p_grid_t * Δt  + c_deg * |p_bat_t| * Δt )
    约束:
        SOC ∈ [SOC_min, SOC_max]；若越界，clip 并附加小惩罚（可选）
    """
    metadata = {"render_modes": []}

    def __init__(self,
                 series,                 # dict from load_power_data()
                 dt_hours=1.0,           # 数据时间步(小时)
                 E_bat_MWh=100.0,        # 电池容量 (MWh)
                 P_bat_max_MW=50.0,      # 电池充放电功率上限 (MW)
                 eta_ch=0.95, eta_dis=0.95,
                 soc_min=0.1, soc_max=0.9, soc_init=0.5,
                 deg_cost_per_MW=0.5,    # 折旧成本系数 ($/MW) —— 与论文“r_BES*|p_BES|”同形
                 penalty_soc=0.0,        # 越界罚（已clip；如需软罚可设小于等于0的额外项）
                 episode_len=None,       # 每个episode的步数（默认用全长）
                 obs_norm=True,
                 # 发电机（DG）
                 use_gen=True,
                 P_gen_min_MW=0.0, P_gen_max_MW=40.0,
                 gen_lin_cost_per_MWh=15.0,  # 简化：燃料成本 $/MWh_e
                 ramp_up_MW_per_h=None,  # 例：5.0；None 表示不启用
                 ramp_dn_MW_per_h=None,
                    ):
        super().__init__()
        self.data = series
        self.N = len(self.data['P'])
        self.dt = float(dt_hours)
        self.E = float(E_bat_MWh)
        self.Pmax = float(P_bat_max_MW)
        self.eta_ch = float(eta_ch)
        self.eta_dis = float(eta_dis)
        self.soc_min, self.soc_max = float(soc_min), float(soc_max)
        self.soc_init = float(soc_init)
        self.lambda_soc = 0.5 * float(np.mean(self.data["P"])) * self.E
        self.deg_c = float(deg_cost_per_MW)
        self.penalty_soc = float(penalty_soc)
        self.obs_norm = obs_norm
        #市场
        self.pi_min = 0.0  # 价格下界（¥/MWh）
        self.pi_max = 1000.0  # 上界，先放宽
        self.pi = 400.0  # 初始价
        self.alpha = 5.0  # tâtonnement 步长（如果你做价格迭代版）
        self.grid_price = 450.0  # 外部电网兜底价（也可时变）
        #锅炉
        self.Fbmax = 10.0
        self.ba = 0.0851
        self.bb = 0.0079
        self.Hbfw = 642.12
        self.Hss = 3407.34
        self.LHV = 25000  # 燃料低位发热值
        self.cco2 = 0.01
        self.use_gen = bool(use_gen)
        self.Pg_min = float(P_gen_min_MW)
        self.Pg_max = float(P_gen_max_MW)
        self.cg_lin = float(gen_lin_cost_per_MWh)
        self.ramp_up = None if ramp_up_MW_per_h is None else float(ramp_up_MW_per_h)
        self.ramp_dn = None if ramp_dn_MW_per_h is None else float(ramp_dn_MW_per_h)
        #排放成本
        self.cf = 45.7
        self.gf = 0.75
        # 归一化系数（简单 MinMax/Std，避免训练不稳定）
        self._L_scale = max(1.0, np.percentile(self.data['L'], 99))
        self._R_scale = max(1.0, np.percentile(self.data['R'], 99))
        self._P_scale = max(1.0, np.percentile(self.data['P'], 99))

        # 空间
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -10.0, self.soc_min, -1.0, -1.0], dtype=np.float32),
            high=np.array([2.0, 2.0, 10000.0, self.soc_max, 1.0, 1.0], dtype=np.float32)
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # episode 滑窗起点
        self.episode_len = int(episode_len) if episode_len else self.N
        self._t0 = 0
        self._t = 0
        self._soc = self.soc_init

    # —— 工具函数 ——
    def _obs(self, t):
        L = self.data['L'][t] / self._L_scale if self.obs_norm else self.data['L'][t]
        R = self.data['R'][t] / self._R_scale if self.obs_norm else self.data['R'][t]
        P = self.data['P'][t] / self._P_scale if self.obs_norm else self.data['P'][t]
        obs = np.array([L, R, P, self._soc, self.data['sin_h'][t], self.data['cos_h'][t]],
                       dtype=np.float32)
        return obs

    def _idx(self, k):  # episode 内的绝对索引
        return self._t0 + k

    # —— Gym 接口 ——
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 你也可以随机抽一段窗口：这里默认整段/顺序
        self._t0 = 0
        self._t = 0
        self.p_gen_prev = 0.0
        self._soc = np.clip(self.soc_init, self.soc_min, self.soc_max)
        return self._obs(self._idx(self._t)), {}

    def step(self, action):
        # 1) 动作缩放到功率
        a = float(np.clip(action[0], -1.0, 1.0))
        p_bat_raw = a * self.Pmax
        p_bat_raw = float(np.clip(p_bat_raw, -self.Pmax, self.Pmax))

        # === 基于 SOC 的“可行功率区间”投影（关键）===
        # 记号：p_bat>0 放电；p_bat<0 充电；dt=小时；E=MWh
        soc = self._soc
        # 还能“安全放电”的最大功率（不越过 soc_min）
        # 由 soc_next >= soc_min 推出：p_bat <= (soc - soc_min) * E * eta_dis / dt
        p_dis_feasible_max = max(0.0, (soc - self.soc_min) * self.E * self.eta_dis / self.dt)

        # 还能“安全充电”的最大（绝对值）功率（不越过 soc_max）
        # 由 soc_next <= soc_max 推出：-p_bat <= (soc_max - soc) * E / (eta_ch * dt)
        p_chg_feasible_max = max(0.0, (self.soc_max - soc) * self.E / (self.eta_ch * self.dt))

        # 由此得到基于 SOC 的功率上下界
        lb_soc = -min(self.Pmax, p_chg_feasible_max)  # 最小可行功率（充电为负）
        ub_soc = min(self.Pmax, p_dis_feasible_max)  # 最大可行功率（放电为正）

        # 把动作投影到 [lb_soc, ub_soc]
        p_bat = float(np.clip(p_bat_raw, lb_soc, ub_soc))

        # 锅炉（两台
        bfw_1 = float(np.clip(action[1], -1.0, 1.0))
        bfw_2 = float(np.clip(action[2], -1.0, 1.0))
        fuel1_kgph, p_gen1 = self._boiler_block(bfw_1)
        fuel2_kgph, p_gen2 = self._boiler_block(bfw_2)
        fuel_kgph = fuel1_kgph + fuel2_kgph
        p_gen = p_gen1 + p_gen2

        # 运营成本（$/h）
        ope = self.cf * fuel_kgph
        emiss = self.gf * p_gen

        cost_gen = ope + self.cco2 * emiss

        # 2) 能量平衡与SOC更新
        t_abs = self._idx(self._t)
        L, R, P = self.data['L'][t_abs], self.data['R'][t_abs], self.data['P'][t_abs]

        # SOC 动力学
        if p_bat < 0:  # 充电
            delta_soc = (self.eta_ch * (-p_bat) * self.dt) / self.E
        else:  # 放电
            delta_soc = -((p_bat) * self.dt / self.E) / self.eta_dis
        soc_next = self._soc + delta_soc

        p_grid = L - R - p_bat - p_gen
        if p_grid < 0:
            p_grid = 0

        # Clip + 软罚（可选）
        over = 0.0
        if soc_next < self.soc_min:
            over = self.soc_min - soc_next
            soc_next = self.soc_min
        elif soc_next > self.soc_max:
            over = soc_next - self.soc_max
            soc_next = self.soc_max

        # 3) 成本与奖励（论文式：购电成本 + BES 折旧）  r = -(cost)
        cost_grid = P * p_grid * self.dt           # $/MWh * MW * h
        cost_deg  = self.deg_c * abs(p_bat) * self.dt
        penalty   = self.penalty_soc * over
        reward = -(cost_grid + cost_deg + cost_gen) + penalty

        # 4) 推进时间
        self._soc = float(soc_next)
        self._t += 1
        done = (self._t >= self.episode_len) or (self._idx(self._t) >= self.N-1)
        info = {
            "p_bat": p_bat,
            "p_grid": p_grid,
            "G_demand": L, "newpower_gen": R, "grid_price": P,"bioler_gen": p_gen,
            "soc": self._soc,
            "cost_grid": cost_grid, "cost_deg": cost_deg ,
            "cost_gen": cost_gen,
        }
        obs_next = self._obs(self._idx(self._t if not done else self._t-1))
        return obs_next, float(reward), done, False, info

        # ================== 单元函数 ==================
        # 锅炉 —— 式(1)(2)

    def _boiler_block(self, M_bfw: float):
        """
        输入：给水流量 M_bfw (kg/h)
        返回：fuel_kgph, heat_kW, nb
        fuel_kgph = M_bfw*(Hss - Hbfw) / (LHV * nb)
        heat_kW   = M_bfw*(Hss - Hbfw) / 3600
        """
        M_bfw = ((M_bfw + 1) / 2) * self.Fbmax
        Hfuel = M_bfw *self.LHV
        #高压蒸汽用于发电
        g_out = ( Hfuel /3.6 ) /1000 * 0.43
        return M_bfw, g_out,

    def render(self):
        pass
