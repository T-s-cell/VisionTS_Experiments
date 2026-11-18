# EXP-046:VisionTS 历史特征诊断(验证集,H=96,四方法对照)

状态:**已冻结**(2026-11-19 评审通过,v3 终版;冻结分支 `repro/ltsf-t46-histdiag`)| 十门禁 ALL PASS(verify_t46.json,G1–G10)| 审计包 `temp/VisionTS_t46_histdiag_20261118_v3.tar.gz` sha256 `626bacba…c9fbd53`(v1 bc2650d8 / v2 b3118c95 作废)| eta 与本地产物(含 16 张案例 png)md5 全同。

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
#       + uni2ts==1.1.1 + statsforecast==1.7.8 + pyarrow 14.0.2(numpy 1.26 pin)
#       + matplotlib;freeze 见 env_freeze_t46diag.txt)
# 权重:HF_HOME=/dev_data/wlt/hf_cache(hf-mirror 下载,model.safetensors
#       sha256 8e3c2451ec14d430bcf45d5d312b68464fed4c654837fa0e76c1ab633fd48a05)
cd /dev_data/wlt/VisionTS_Experiments/long_term_tsf/repro_t46_histdiag
HF_HOME=/dev_data/wlt/hf_cache \
WORKERS=32 TIMEOUT=1800 \
bash run_t46_all.sh /dev_data/wlt/VisionTS_Experiments/long_term_tsf
```

分步等价:`select_t46.py`(冻结清单,重跑 sha 一致)→ `preflight_t46.py`(版本/签名/权重/arr_val 预检)→ `infer_sfdx_t46.py --selftest`(超时真杀+管线存活验证)→ `infer_vts_t46.py`(切片+全 1792 真值核验)→ 冒烟 `--smoke`(ETTh1/ETTm1 各 8;moirai/sfdx/collect)→ 全量(moirai/sfdx)→ `build_features_t46.py` → `collect_metrics_t46.py` → `analyze_t46.py` → `figures_t46.py` → `gen_report_numbers.py`(报告唯一数字来源)→ `verify_t46.py`(十门禁)。

CPU 池:持久 worker 进程(数值库线程=1/进程);超时 terminate+kill 真杀,该样本记 timeout,池补位继续;个别样本失败入 `failure_ledger.csv` 不中止,管线/对齐错误才中止。**实测调整记录**:冒烟(8 worker/300s)AutoARIMA ETTh1 4/8、ETTm1 7/8 超时(P=96 季节搜索慢,ok 样本耗时 45–284s)→ 全量按"实测后调整"改为 32 worker/1800s;AutoETS 冒烟 8/8 ok(~0.2s/样本)。

## 分析口径(冻结于 feature_formulas.md,分析前落盘)

- 特征(仅输入历史):period_corr、recent_old、trend、vol、hist_std;near_constant(hist_std<1e-8)单独标记,不进 tertile;
- tertile:每 (ds×变量) 内 64 非近常数样本按特征 1/3、2/3 分箱 low/mid/high;
- 相对 gap = (MSE_VTS−MSE_B)/(MSE_VTS+MSE_B+1e-12),正 = VisionTS 相对弱势线索;
- **单元级**方向 = 每 (ds×var) 内 tertile low/mid/high 的 mean rel_gap;
- **变量级**方向(v2 显式口径)= 每 ds 先取(high 单元 rel_gap − low 单元 rel_gap),再跨 4 个 ds 取中位(`direction_by_var_t46.csv`);
- **幅度分层**(v2)= 同一 (ds×var) 单元内:切分阈值取**该单元全部非近常数样本**的目标特征中位数(先于分层确定,不随幅度层重算),在 hist_std 三分位层内应用该阈值;各侧取 common-ok 样本,rel_gap 由**各侧组均 MSE** 代入 (a−b)/(a+b+1e-12)(非逐样本 gap 的均值);两侧均非空才计 delta,一侧为空的比较丢弃(有效比较 749/1008 行;部分比较一侧仅 **1–4 个样本**),delta 跨 (ds×var) 单元取中位(`amplitude_strat_t46.csv` / `_summary_`);
- 共同样本:两两比较用双方均成功;四方法排名用四者共同;覆盖率细分到特征组;
- 共同困难:每 (ds×var×方法) 各自 ok 样本 90 分位阈值,四者共同成功样本全部达标判共同困难,允许空;
- 案例:明显较好 = VTS 排名 1 且 MSE<0.8×次优;明显较差 = 排名 4 且 MSE>1.25×次差(margin=VTS/次差);每类每 ds margin 前 2,不足则空缺;
- 报告数字全部由 `gen_report_numbers.py` 自动生成(`report_numbers_t46.json` / `report_tables_t46.md`),禁止手抄。

## 校验门禁(verify_t46.json,十项 ALL PASS)

G1 清单 sha+字节重放;G2 全 1792 真值/对齐;G3 覆盖率↔台账↔npz 三方一致;G4 overall 重算 ≤1e-9;G5 有限性;G6 item_id↔起点;**G7 特征逐样本复算(≤1e-8)+ 三分位分箱计数复算;G8 direction_summary/direction_by_var 独立重算(≤1e-9);G9 案例规则重算(id 与 margin 逐项一致);G10 幅度分层逐行重算(≤1e-9)**。

## 输出文件(save_t46_histdiag/)

sample_list_t46.csv(+sha)、config_t46.json(preflight)、arr_vts_t46.npz、arr_moirai[_smoke]_<DS>.npz、AutoETS/AutoARIMA[_smoke]_<DS>.npz、moirai_meta.json、sfdx_meta.json、failure_ledger.csv、features_t46.csv、per_sample_metrics[_smoke].csv、collect_meta.json、overall_t46.csv、table_tertile_t46.csv、coverage_by_group_t46.csv、direction_summary_t46.csv、direction_by_var_t46.csv、amplitude_strat_t46.csv、amplitude_strat_summary_t46.csv、joint_hard_t46.csv、joint_hard_summary_t46.csv、cases_t46.csv、cases_png/(better_*/worse_* 共 16 张)、fig_tertile_summary.png、analyze_meta_t46.json、report_numbers_t46.json、report_tables_t46.md、verify_t46.json。审计包另含脚本目录下的 `features_t46.csv`(与 build_features 产物逐字节一致)。

## 必答三问(报告 report_t46.md)

1. 差距是否随某历史特征稳定变化(tertile 单调 + 方向跨 ds/变量一致)?
2. 是否跨变量出现?
3. 是否只是所有方法共同变难(对照误差同升而 gap 平 → 样本难;gap 随特征扩大 → VTS 相对弱势线索)?

无稳定变化或仅个别变量 → 如实报告,不据此支持机制假设。本轮完成后停止;扩展(Weather/Electricity/其他 H/干预)待用户决定。
