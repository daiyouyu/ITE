import pandas as pd
import numpy as np
from typing import Optional,Dict

def load_power_data(
        csv_path: str,
                    start=None, end=None):
    df = pd.read_csv(csv_path, parse_dates=['date'])
    if start is not None: df = df[df['date'] >= pd.to_datetime(start)]
    if end   is not None: df = df[df['date'] <= pd.to_datetime(end)]
    df = df.sort_values('date').reset_index(drop=True)

    # 负荷 (系统级): 8个分区之和
    load_cols = ['WEST','SOUTH','EAST','FWEST','SCENT','NORTH','NCENT','COAST']
    #load_cols = ['COAST',  'NCENT', 'SOUTH', 'WEST']
    L = df[load_cols].astype(float).values  # MW
    # 可再生 (系统级): 风 + 光
    R_wind  = df['WIND_ACTUAL_SYSTEM_WIDE'].astype(float).values
    R_solar = df['SOLAR_ACTUAL_SYSTEM_WIDE'].astype(float).values  # MW
    R = [R_wind * 0.0  + R_solar * 1.5,
         R_wind * 0.5 + R_solar * 0,
         R_wind * 0.0  + R_solar * 1.5,
         R_wind * 0.25 + R_solar * 0,]
    #R = [R * 0.3, R * 0.3, R * 0.05, R * 0.1, R * 0.1,R * 0.1,R *0.1,R* 0.1]
    # 电价：均值或指定LZ_*列
    lz_cols = ['LZ_AEN','LZ_CPS','LZ_HOUSTON','LZ_LCRA','LZ_NORTH','LZ_RAYBN','LZ_SOUTH','LZ_WEST']
    P = df[lz_cols].astype(float).values  # MWh

    # 时间特征（小时正余弦）
    hours = df['date'].dt.hour.values
    sin_h = np.sin(2*np.pi*hours/24.0)
    cos_h = np.cos(2*np.pi*hours/24.0)

    data = []
    for i in range(4):
        data.append({
            'datetime': df['date'].values,
            'L': L[:,i],
            'R': R[i]*0.3,
            'P': P[:,i],
            'sin_h': sin_h,
            'cos_h': cos_h
        })
    return data

def load_ITE_data(
    G_csv_path: str,
    H_csv_path: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    out_freq: str = "1h",         # 目标时间分辨率：'1H'（小时）；若想保持15分钟，设为'15min'
):
    # ---------- 1) 读取并按时间裁剪 ----------
    tcol = "Time"

    g_df = pd.read_csv(G_csv_path, parse_dates=['Time']).sort_values(tcol)
    h_df = pd.read_csv(H_csv_path, parse_dates=['Time']).sort_values(tcol)

    if start is not None:
        start = pd.to_datetime(start)
        g_df = g_df[g_df[tcol] >= start]
        h_df = h_df[h_df[tcol] >= start]
    if end is not None:
        end = pd.to_datetime(end)
        g_df = g_df[g_df[tcol] <= end]
        h_df = h_df[h_df[tcol] <= end]

    # 设为索引，先确保 15min 规则性，再重采样到 out_freq
    g_df = g_df.set_index(tcol).sort_index()
    h_df = h_df.set_index(tcol).sort_index()

    # ---------- 2) 定义分区列 ----------
    # 电负荷分区（与你原函数一致）
    elec_res_cols = ['N105','N106','N107','N109','N110','N111','N112','N113','N114',
                     'N116','N118','N125','N126','N127','N128','N129','N130']
    elec_ind_cols = ['N83','N84']
    elec_com_cols = ['N103','N122','N85','N86','N87','N88','N89','N90','N91','N92',
                     'N93','N94','N95','N96','N97','N98']
    elec_pub_cols = ['N76','N77','N78','N79','N80','N81','N104','N108','N115','N117',
                     'N120','N121','N123','N124','N99','N100','N101','N102']

    # 热负荷分区（与你原函数一致）
    heat_res_cols = elec_res_cols
    heat_ind_cols = ['N84']
    heat_com_cols = elec_com_cols
    heat_pub_cols = elec_pub_cols

    # 可再生列（热表里）
    wind_cols  = ['WT1 (MW)','WT2 (MW)','WT3 (MW)','WT4 (MW)','WT5 (MW)']
    solar_cols = ['PV1 (MW)','PV2 (MW)','PV3 (MW)','PV4 (MW)','PV5 (MW)',
                  'PV6 (MW)','PV7 (MW)','PV8 (MW)','PV9 (MW)','PV10 (MW)',
                  'PV11 (MW)','PV12 (MW)','PV13 (MW)','PV14 (MW)','PV15 (MW)']

    # ---------- 3) 数值化 + 小缺口插值 ----------
    def to_float(df, cols):
        exists = [c for c in cols if c in df.columns]
        return df[exists].apply(pd.to_numeric, errors='coerce')

    # 先把需要的列抽出来并数值化
    g_elec_res = to_float(g_df, elec_res_cols)
    g_elec_ind = to_float(g_df, elec_ind_cols)
    g_elec_com = to_float(g_df, elec_com_cols)
    g_elec_pub = to_float(g_df, elec_pub_cols)

    h_heat_res = to_float(h_df, heat_res_cols)
    h_heat_ind = to_float(h_df, heat_ind_cols)
    h_heat_com = to_float(h_df, heat_com_cols)
    h_heat_pub = to_float(h_df, heat_pub_cols)

    h_wind  = to_float(h_df, wind_cols)
    h_solar = to_float(h_df, solar_cols)

    # 电价
    price_series = pd.to_numeric(g_df["price_per_kWh"], errors='coerce')

    # 可选：对 15min 小缺口做前向插值（不改变统计量）
    for _df in [g_elec_res,g_elec_ind,g_elec_com,g_elec_pub,
                h_heat_res,h_heat_ind,h_heat_com,h_heat_pub,
                h_wind,h_solar]:
        _df.interpolate(limit=4, limit_direction='forward', inplace=True)

    price_series = price_series.interpolate(limit=4, limit_direction='forward')

    # ---------- 4) 重采样到小时（均值） ----------
    agg = "mean"  # 功率/价格取均值；若要电量，后续乘以持续时间
    g_res_1H = g_elec_res.resample(out_freq).agg(agg).sum(axis=1)
    g_ind_1H = g_elec_ind.resample(out_freq).agg(agg).sum(axis=1)
    g_com_1H = g_elec_com.resample(out_freq).agg(agg).sum(axis=1)
    g_pub_1H = g_elec_pub.resample(out_freq).agg(agg).sum(axis=1)

    h_res_1H = h_heat_res.resample(out_freq).agg(agg).sum(axis=1)
    h_ind_1H = h_heat_ind.resample(out_freq).agg(agg).sum(axis=1)
    h_com_1H = h_heat_com.resample(out_freq).agg(agg).sum(axis=1)
    h_pub_1H = h_heat_pub.resample(out_freq).agg(agg).sum(axis=1)

    wind_1H  = h_wind.resample(out_freq).agg(agg).sum(axis=1)
    solar_1H = h_solar.resample(out_freq).agg(agg).sum(axis=1)
    r_1H = wind_1H.add(solar_1H, fill_value=0.0)

    price_1H = None
    if price_series is not None:
        price_1H = price_series.resample(out_freq).mean()

    # 对齐索引（取交集，避免缺失）
    idx = g_res_1H.index
    for s in [g_ind_1H,g_com_1H,g_pub_1H,
              h_res_1H,h_ind_1H,h_com_1H,h_pub_1H,
              r_1H]:
        idx = idx.intersection(s.index)
    if price_1H is not None:
        idx = idx.intersection(price_1H.index)

    g_res_1H = g_res_1H.reindex(idx)
    g_ind_1H = g_ind_1H.reindex(idx)
    g_com_1H = g_com_1H.reindex(idx)
    g_pub_1H = g_pub_1H.reindex(idx)

    h_res_1H = h_res_1H.reindex(idx)
    h_ind_1H = h_ind_1H.reindex(idx)
    h_com_1H = h_com_1H.reindex(idx)
    h_pub_1H = h_pub_1H.reindex(idx)

    r_1H = r_1H.reindex(idx)
    if price_1H is not None:
        price_1H = price_1H.reindex(idx)

    # ---------- 5) 组织输出 ----------
    # 电负荷矩阵 (4, T)
    G_L = np.vstack([
        g_res_1H.values,
        g_ind_1H.values,
        g_com_1H.values,
        g_pub_1H.values
    ])

    # 热负荷矩阵 (4, T)
    H_L = np.vstack([
        h_res_1H.values,
        h_ind_1H.values,
        h_com_1H.values,
        h_pub_1H.values
    ])

    # 可再生功率 (4, T) —— 你原来等比复制
    R = np.vstack([r_1H.values*0.5,
                   r_1H.values*0.5,
                   r_1H.values*0.5,
                   r_1H.values*0.5])

    # 电价 (4, T)
    if price_1H is not None:
        P = np.vstack([price_1H.values,
                       price_1H.values,
                       price_1H.values,
                       price_1H.values])
    else:
        P = np.full((4, len(idx)), np.nan, dtype=float)

    # 时间特征
    hours = idx.hour.values
    sin_h = np.sin(2*np.pi*hours/24.0)
    cos_h = np.cos(2*np.pi*hours/24.0)

    data = []
    for i in range(4):
        data.append({
            'datetime': idx.to_numpy(),
            'H_L'     : H_L[i, :],     # MWth，小时均值
            'G_L'     : G_L[i, :],     # MW，小时均值
            'R'       : R[i, :],       # MW，小时均值
            'P'       : P[i, :] * 1000,       # 元/MWh，小时均值
            'sin_h'   : sin_h,
            'cos_h'   : cos_h
        })
    return data

