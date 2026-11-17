# EXP-046:VisionTS 历史特征诊断(验证集,H=96,四方法对照)

目的:观察 VisionTS 在哪些输入历史特征下相对对照方法变差;区分「VisionTS 相对弱势线索」与「样本本身难(各方法共同变难)」。只发现线索,不支持机制结论;窗口重叠,1792 样本仅描述性分析。

## 方法

| 臂 | 来源 | 说明 |
|---|---|---|
| vts | EXP-043 `save_t43_period_val_val/<DS>_96_P<P>/arr_val.npz` 切片 | 完全冻结,零推理;arr_index = start_row − 12K |
| moirai | uni2ts 1.1.1,Salesforce/moirai-1.0-R-base | patch_size=32(**尚未验证是否最优**,本轮预设)、num_samples=200、逐时刻中位数;种子 2021(random/numpy/torch/cuda) |
| autoets / autoarima | statsforecast 1.7.8 | 逐样本窗口内拟合,season_length=P,其余默认(签名/默认值快照进 meta);**单列标注,不称冻结模型** |

统一:标准化尺度(train [0,12K) StandardScaler ddof=0)、同历史(L)/同变量/同起点;真值 = csv 标准化 f64 重算(数组 f32,容差 1e-5)。

## 复现命令(eta)

```bash
# 环境:/dev_data/wlt/conda/envs/t46diag(python3.11 + torch 2.4.1+cu121
#       + uni2ts==1.1.1 + statsforecast==1.7.8;freeze 见 env_freeze_t46diag.txt)
cd /dev_data/wlt/VisionTS_Experiments/long_term_tsf/repro_t46_histdiag
HF_HOME=/dev_data/wlt/hf_cache \
WORKERS=8 TIMEOUT=300 \
bash run_t46_all.sh /dev_data/wlt/VisionTS_Experiments/long_term_tsf
```

分步等价:`select_t46.py`(冻结清单,重跑 sha 一致)→ `preflight_t46.py`(版本/签名/权重/arr_val 预检)→ `infer_sfdx_t46.py --selftest`(超时真杀+管线存活验证)→ `infer_vts_t46.py`(切片+全 1792 真值核验)→ 冒烟 `--smoke`(ETTh1/ETTm1 各 8;moirai/sfdx/collect)→ 全量(moirai/sfdx)→ `build_features_t46.py` → `collect_metrics_t46.py` → `analyze_t46.py` → `figures_t46.py` → `verify_t46.py`(六门禁)。

CPU 池:持久 worker 进程(默认 8,数值库线程=1/进程);超时(默认 300s)terminate+kill 真杀,该样本记 timeout,池补位继续;个别样本失败入 `failure_ledger.csv` 不中止,管线/对齐错误才中止。

## 分析口径(冻结于 feature_formulas.md,分析前落盘)

- 特征(仅输入历史):period_corr、recent_old、trend、vol;near_constant(hist_std<1e-8)单独标记,不进 tertile;
- tertile:每 (ds×变量) 内 64 非近常数样本按特征 1/3、2/3 分箱 low/mid/high;
- 相对 gap = (MSE_VTS−MSE_B)/(MSE_VTS+MSE_B+1e-12),正 = VisionTS 相对弱势线索;
- 共同样本:两两比较用双方均成功;四方法排名用四者共同;覆盖率细分到特征组;
- hist_std 为第 5 分组维(描述性,检验幅度解释);
- 共同困难:每 (ds×var×方法) 各自 ok 样本 90 分位阈值,四者共同成功样本全部达标判共同困难,允许空;
- 案例:明显较好 = VTS 排名 1 且 MSE<0.8×次优;明显较差 = 排名 4 且 MSE>1.25×次差;每类每 ds margin 前 2,不足则空缺。

## 输出文件(save_t46_histdiag/)

sample_list_t46.csv(+sha)、config_t46.json(preflight)、arr_vts_t46.npz、arr_moirai[_smoke]_<DS>.npz、AutoETS/AutoARIMA[_smoke]_<DS>.npz、moirai_meta.json、sfdx_meta.json、failure_ledger.csv、features_t46.csv、per_sample_metrics[_smoke].csv、collect_meta.json、overall_t46.csv、table_tertile_t46.csv、coverage_by_group_t46.csv、direction_summary_t46.csv、joint_hard_t46.csv、joint_hard_summary_t46.csv、cases_t46.csv、cases_png/、fig_tertile_summary.png、analyze_meta_t46.json、verify_t46.json。

## 必答三问(报告 report_t46.md)

1. 差距是否随某历史特征稳定变化(tertile 单调 + 方向跨 ds/变量一致)?
2. 是否跨变量出现?
3. 是否只是所有方法共同变难(对照误差同升而 gap 平 → 样本难;gap 随特征扩大 → VTS 相对弱势线索)?

无稳定变化或仅个别变量 → 如实报告,不据此支持机制假设。本轮完成后停止;扩展(Weather/Electricity/其他 H/干预)待用户决定。
