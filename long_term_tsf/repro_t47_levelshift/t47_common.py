"""EXP-047 旧历史水平对预测的影响验证(level-shift):公共常量/条件表/构造公式。

复用 EXP-046(冻结头 21e2fa3)公共函数,单一来源不复制逻辑:
    from t46_common import … (DS_CFG/load_csv/train_scaler/std_all/get_window/…)

条件表(用户规格 2026-11-19):
- 基准窗口:EXP-046 清单每 (ds×var) recent_old 最低 16 个 → 448;
- δ ∈ s·{-1,-0.5,+0.5,+1},s = x.std(ddof=0)(csv f64 重算);
- m/P ∈ {1,2,4}(新水平已持续长度);
- A 组(参照):x_A = x + δ(整段);B 组(冲突):仅末 m 步 + δ;
- 真值 y' = y + δ(A/B 共享;A/B 末 m 步逐位相同);
- cond_id:A 组 = delta_idx(0..3);B 组 = 4 + 3*delta_idx + m_idx(m_idx 0..2 ↔ m/P=1,2,4);
  (sid_t46, cond_id) 唯一;δ=0 仅冒烟,不计入 16 条件;
- 行序 = window-major then cond:row_global = w_global*16 + cond_id,
  w_global 按 ds(DS_ORDER)→ var(channel 序)→ rank(recent_old 升序)。
"""
import os
import sys

_T46 = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', 'repro_t46_histdiag'))
sys.path.insert(0, _T46)
from t46_common import (CHANNELS, DS_CFG, DS_ORDER, H, NEAR_CONST_EPS, SEED, TRUTH_TOL,  # noqa: F401
                        get_window, hist_features, load_csv, sample_list_path, sha256_file,
                        std_all, train_scaler)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
T46_DIR = _T46
CKPT_SHA = 'c551c37b63a221a3e6d7d7f49e96f34078c9b183f788bedc0387493521bade0e'  # mae_visualize_vit_base.pth

DELTA_MULTS = [-1.0, -0.5, 0.5, 1.0]
M_MULTS = [1, 2, 4]
N_CONDS = 16  # A4 + B12
METHODS = ['vts', 'moirai', 'autoets', 'autoarima']
MAIN_METHODS = ['vts', 'moirai', 'autoets']  # 全 7168 条件
S_TOL = 1e-8       # 构造重算 s vs features_t46.csv hist_std
SMOKE_TOL_T43 = 1e-5   # 冒烟 S1-a:t43 原组织复现 vs arr_vts_t46
SMOKE_TOL_NEW = 5e-4   # 冒烟 S1-b:t47 单变量路径 vs S1-a 结果


def cond_table():
    """[(cond_id, group, delta_idx, m_idx)] 恰 16 条;m_idx None 表 A 组。"""
    rows = [(di, 'A', di, None) for di in range(4)]
    rows += [(4 + 3 * di + mi, 'B', di, mi) for di in range(4) for mi in range(3)]
    rows.sort()
    assert [r[0] for r in rows] == list(range(N_CONDS))
    return rows


CONDS = cond_table()


def cond_params(cond_id):
    """cond_id → (group, delta_idx, m_idx, delta_mult, m_over_P)。A 组 m_over_P=None。"""
    g, di, mi = CONDS[cond_id][1], CONDS[cond_id][2], CONDS[cond_id][3]
    return g, di, mi, DELTA_MULTS[di], (None if mi is None else M_MULTS[mi])


def apply_variant(x, delta, m_steps, group):
    """构造输入:f64 不就地板复制后加 δ。A:整段;B:仅末 m_steps。"""
    x2 = x.copy()
    if group == 'A':
        x2 += delta
    else:
        assert 0 < m_steps <= x.shape[0]
        x2[-m_steps:] += delta
    return x2


def baseline_list_path():
    return os.path.join(SCRIPT_DIR, 'baseline_list_t47.csv')


def manifest_path():
    return os.path.join(SCRIPT_DIR, 'manifest_t47.csv')


SMOKE_DS = ('ETTh1', 'ETTm1')  # 冒烟数据集:各取前 2 个基准窗口(清单行序)


def smoke_subset(man):
    """manifest → 冒烟行(每冒烟 ds 前 2 个基准窗 ×16 条件 = 64 行,保序)。"""
    out = []
    for ds in SMOKE_DS:
        rows = [r for r in man if r['ds'] == ds]
        sids = []
        for r in rows:  # manifest 序 = window-major then cond
            if r['sid_t46'] not in sids:
                sids.append(r['sid_t46'])
            if len(sids) == 2:
                break
        assert len(sids) == 2, (ds, len(sids))
        out.extend(r for r in rows if r['sid_t46'] in sids)
    assert len(out) == 64, len(out)
    return out
