"""EXP-046 VisionTS 历史特征诊断:公共常量/采样/标准化/特征公式。

口径依据(2026-11-18 实测,与资料包 v3 勘误一致):
- ETTh/ETTm 划分 border2s = [8640,11520,14400] / [34560,46080,57600](12/4/4 月);
- val 段预测区 = [12K,16K),H=96 合法预测起点(预测区起点行)= [12K, 16K-H];
- EXP-043 val 臂 arr_val.npz 仅 pred/true,f32 (N,96,7),窗 i 预测区 = [12K+i, 12K+i+96)
  → arr_index = start_row - 12K;
- StandardScaler 仅在 train [0,12K) fit,numpy std ddof=0(TSLib 语义);
- 变量序 = csv 列序 [HUFL,HULL,MUFL,MULL,LUFL,LULL,OT](OT 末位)。
"""
import csv
import os

import numpy as np

H = 96
SEED = 2021
N_SEG = 64
NEAR_CONST_EPS = 1e-8  # 标准化尺度下窗口 std 低于此判近常数
TRUTH_TOL = 1e-5  # 数组 true(f32)vs csv 标准化 f64 重算的核验容差(f32 链路实测 ~2e-7)

DS_ORDER = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']
CHANNELS = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

DS_CFG = {
    'ETTh1': dict(csv='dataset/ETT-small/ETTh1.csv', K=720, L=2880, P=24,
                  group='ETTh1_96_P24', freq='h', csv_rows=17420),
    'ETTh2': dict(csv='dataset/ETT-small/ETTh2.csv', K=720, L=1728, P=24,
                  group='ETTh2_96_P24', freq='h', csv_rows=17420),
    'ETTm1': dict(csv='dataset/ETT-small/ETTm1.csv', K=2880, L=2304, P=96,
                  group='ETTm1_96_P96', freq='15T', csv_rows=69680),
    'ETTm2': dict(csv='dataset/ETT-small/ETTm2.csv', K=2880, L=4032, P=96,
                  group='ETTm2_96_P96', freq='15T', csv_rows=69680),
}

SAMPLE_LIST = 'sample_list_t46.csv'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def sample_list_path():
    """清单固定放脚本目录(与代码一起 rsync/版本化),不随 repo_root/cwd 漂移。"""
    return os.path.join(SCRIPT_DIR, SAMPLE_LIST)


def load_csv(csv_path):
    """读 ETT csv → (T,7) float64,丢 date 列;校验行数与列数。"""
    with open(csv_path) as f:
        rd = csv.reader(f)
        header = next(rd)
        rows = [[float(x) for x in r[1:]] for r in rd]
    data = np.asarray(rows, dtype=np.float64)
    assert len(header) == 8 and header[1:] == CHANNELS, header
    return data


def train_scaler(data, K):
    """train [0,12K) 段 mean/std(ddof=0)。"""
    tr = data[:12 * K]
    return tr.mean(0), tr.std(0)


def std_all(data, mean, scale):
    return (data - mean) / scale


def legal_starts(K):
    """H=96 时合法预测区起点行(闭区间 [12K, 16K-H])。"""
    return np.arange(12 * K, 16 * K - H + 1)


def sample_starts(K, rng):
    """64 段各随机 1 起点;调用序 = DS_ORDER,共用同一 rng(确定性)。"""
    segs = np.array_split(legal_starts(K), N_SEG)
    assert len(segs) == N_SEG
    return [int(seg[int(rng.integers(0, len(seg)))]) for seg in segs]


def iter_samples():
    """采样清单行序:ds → seg → channel;每行 (ds, seg, ch, start_row, arr_index)。"""
    rng = np.random.default_rng(SEED)
    sid = 0
    for ds in DS_ORDER:
        cfg = DS_CFG[ds]
        starts = sample_starts(cfg['K'], rng)
        for seg, s in enumerate(starts):
            for ch in range(7):
                yield sid, ds, seg, ch, s, s - 12 * cfg['K']
                sid += 1


def get_window(std, ds, start_row, ch):
    """标准化输入窗 (L,) float64;历史 = [start-L, start)。"""
    L = DS_CFG[ds]['L']
    assert start_row - L >= 0
    return std[start_row - L:start_row, ch]


def hist_features(x, P):
    """仅由输入历史计算的特征(标准化窗 x,L≥4P 必须)。

    返回 dict:hist_std / period_corr / recent_old / trend / vol / near_constant。
    近常数样本特征值记 NaN(除 hist_std),单独标记不进 tertile。
    """
    out = dict(hist_std=float(x.std()), near_constant=False)
    if out['hist_std'] < NEAR_CONST_EPS:
        out['near_constant'] = True
        for k in ('period_corr', 'recent_old', 'trend', 'vol'):
            out[k] = float('nan')
        return out
    xp = x[P:]
    x0 = x[:-P]
    denom = x0.std() * xp.std()
    out['period_corr'] = float(((x0 - x0.mean()) * (xp - xp.mean())).mean() / denom) if denom > 0 else float('nan')
    phase = x[-4 * P:-P].reshape(3, P).mean(0)
    out['recent_old'] = float(np.sqrt(((x[-P:] - phase) ** 2).mean()) / out['hist_std'])
    t = np.arange(4 * P, dtype=np.float64)
    slope = np.polyfit(t, x[-4 * P:], 1)[0]
    out['trend'] = float(abs(slope) * (4 * P) / out['hist_std'])
    d = np.diff(x)
    out['vol'] = float(d.std() / out['hist_std'])
    return out


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()
