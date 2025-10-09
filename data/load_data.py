import pandas as pd
import numpy as np

def load_power_data(csv_path: str,
                    price_mode: str = "mean",   # 'mean' 或 指定某个LZ列名，如 'LZ_HOUSTON'
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
    R = [R_wind * 0.25  + R_solar * 0.9,
         R_wind * 0.25 + R_solar * 0,
         R_wind * 0.25  + R_solar * 0.9,
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
            'R': R[i]*0.5,
            'P': P[:,i],
            'sin_h': sin_h,
            'cos_h': cos_h
        })
    return data

