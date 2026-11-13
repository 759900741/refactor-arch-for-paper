# 数据质量与预处理报告（阶段一自动生成）

> 生成时间：2026-11-11 14:50:01 ｜ 滞后特征=True ｜ 归一化=minmax


## 荆门市2024（jingmen）

- 文件：`荆门市2024负荷天气.csv`
- 时间规整：{'n_raw': 35136, 'parse_fail': 0, 'n_dup': 0, 'n_rows_after': 35136, 'freq': '15min', 'added_steps': 0, 'missing_steps': 0}
- 异常检测：{'method': 'global', 'n_sigma': 126, 'n_diff': 318, 'diff_threshold': 178.65781891977514}
- 特征数：29；剔除常量列：[]；丢弃起始NaN行：672
- 划分边界：
    - train: 2024-01-08 00:00:00 ~ 2024-10-21 04:30:00（27571 行）
    - val: 2024-10-21 04:45:00 ~ 2024-11-26 02:00:00（3446 行）
    - test: 2024-11-26 02:15:00 ~ 2024-12-31 23:45:00（3447 行）
- 滑窗（L=96, H=4, stride=4）：
    - train: X=[6868, 96, 29], y=[6868, 4]
    - val: X=[837, 96, 29], y=[837, 4]
    - test: X=[837, 96, 29], y=[837, 4]
- 特征列：负荷, max_temperature, low_temperature, avg_temperature, avg_shidu, jiangyuliang, wind_speed, cloud_cover, surface_pressure, shortwave_radiation, dew_point, is_workday, is_holiday, season, hour, minute, dayofweek, month, dayofyear, is_weekend, hour_sin, hour_cos, dow_sin, dow_cos, doy_sin, doy_cos, day_type, lag_1d, lag_1w

## 兰州市2023（lanzhou）

- 文件：`兰州用电负荷数据.csv`
- 时间规整：{'n_raw': 8760, 'parse_fail': 0, 'n_dup': 0, 'n_rows_after': 8760, 'freq': '1h', 'added_steps': 0, 'missing_steps': 0}
- 异常检测：{'method': 'hampel', 'win': 25, 'k': 3.0, 'n_flagged': 92}
- 特征数：27；剔除常量列：['minute']；丢弃起始NaN行：168
- 划分边界：
    - train: 2023-01-08 00:00:00 ~ 2023-10-21 08:00:00（6873 行）
    - val: 2023-10-21 09:00:00 ~ 2023-11-26 03:00:00（859 行）
    - test: 2023-11-26 04:00:00 ~ 2023-12-31 23:00:00（860 行）
- 滑窗（L=24, H=1, stride=1）：
    - train: X=[6849, 24, 27], y=[6849, 1]
    - val: X=[835, 24, 27], y=[835, 1]
    - test: X=[836, 24, 27], y=[836, 1]
- 特征列：electricity(KW), cold(ton), heat(mmBTU), Temperature, Wind Speed, Wind Direction, Surface Albedo, Pressure, Precipitable Water, Solar Zenith Angle, Relative Humidity, Dew Point, Cloud Type, hour, dayofweek, month, dayofyear, is_weekend, hour_sin, hour_cos, dow_sin, dow_cos, doy_sin, doy_cos, day_type, lag_1d, lag_1w

## 说明与注意

- 荆门采用全局 3σ + 差分阈值标记异常；兰州采用局部 Hampel 滤波（win=25, k=3）标记毛刺，二者均避免误删夏季真实峰/换挡段。
- 异常仅做标记（is_anomaly 列），未删除；是否修正留待阶段二 RPCA。
- 兰州无节假日日历，day_type 以「周一~周五=工作日、周末=周末」近似（泛化集，可接受）。
- 兰州 cold/heat 与温度强共线，已保留并待阶段二 MIV 裁决。
- 归一化仅在训练集 fit，验证/测试仅 transform（防泄漏）。
- 多步（荆门 H=4）采用 stride=H 避免相邻样本目标重叠。