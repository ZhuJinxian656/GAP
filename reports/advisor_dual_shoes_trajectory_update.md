# 给 Advisor 的 Place Dual Shoes 轨迹监督更新

生成时间：2026-06-28

## 1. 复现状态更新

GitHub issue 中也有人报告官方 released code 在 `place_dual_shoes-demo_clean-50` 上约 `11/100`。我当前 vanilla 是 `4/50 = 8%`，与这个外部线索同量级。

因此，我建议当前工作不再定义为论文主表 100-demo reproduction，而是定义为：

> public 50-demo Place Dual Shoes low-success regime 下的 mechanism study。

这个 setting 可以继续作为机制研究环境，但不能和论文主表直接比较。

## 2. 之前实验的意义

之前 9 个 eval50 variant 是 smoke-level mechanism evidence：

| variant | result |
| --- | ---: |
| vanilla | `4/50` |
| dino_only | `8/50` |
| no_future | `1/50` |
| pi3_pooled | `6/50` |
| pi3_compressed | `2/50` |
| pi3_random | `0/50` |
| pi3_dropout | `4/50` |
| pi3_eef_region | `6/50` |
| pi3_non_eef_region | `5/50` |

当前读法是：

- full dense Pi3 解释不稳，因为 `dino_only` 和 `pi3_pooled` 都不差。
- `no_future` 很低，说明 future supervision 仍可能重要。
- `dino_only / pi3_pooled` 不差，说明 dense full-scene Pi3 可能不是唯一关键。
- `pi3_eef_region` 和 `pi3_non_eef_region` 很接近，因此 EEF proxy 还不能支持 object-hand 机制结论。

## 3. 新问题

核心问题从“Pi3 有没有用”变成：

> future Pi3 supervision 到底应该监督完整未来场景，还是监督动作导致的变化？

也就是说，GAP 的 full future Pi3 target 可能过宽。它既包含 action-relevant 的变化，也包含大量与动作弱相关的场景 token。对 bimanual action expert 来说，更合理的 auxiliary target 可能是：

- `z_future - z_current`，即 delta future；
- 只监督 `||z_future - z_current||` 较大的 changed tokens；
- 之后如果有真实 mask，再验证 object-hand / interaction-region 版本。

本轮不声称 object-hand 或 triadic 机制，因为当前数据没有真实 object masks / hand masks / object pose。

## 4. 本轮新增工作

本轮已经完成：

- 生成 eval100 baseline consolidation 命令：
  - `scripts/generate_dual_shoes_eval100_commands.py`
  - `scripts/generated_dual_shoes_eval100_commands.sh`
- 生成结果 parser 和当前表：
  - `scripts/parse_dual_shoes_results.py`
  - `reports/dual_shoes_trajectory_results.json`
  - `reports/dual_shoes_trajectory_results_table.md`
  - `reports/dual_shoes_eval100_existing_baselines.json`
- 增加 future Pi3 copyability diagnostic：
  - `scripts/analyze_future_pi3_copyability.py`
  - `reports/future_pi3_copyability_dual_shoes50.json`
- 增加 action-coupling diagnostic：
  - `scripts/analyze_action_coupling_gain.py`
  - `reports/action_coupling_gain_dual_shoes50.json`
- 实现新 future supervision modes：
  - `future_target_mode=pi3_delta`
  - `future_target_mode=pi3_changed_tokens`
  - `future_target_mode=pi3_delta_changed_tokens`
- 增加 smoke test：
  - `scripts/smoke_test_trajectory_future_modes.py`
- 生成新训练/eval 命令：
  - `scripts/generate_dual_shoes_trajectory_commands.py`
  - `scripts/generated_dual_shoes_trajectory_commands.sh`
- 写入主报告：
  - `reports/dual_shoes_50demo_trajectory_supervision.md`

诊断结果摘要：

- copyability diagnostic 使用 512 个均匀采样 sequence starts：
  - `MSE(z_current, z_future) = 0.016599`
  - token delta norm p50 = `1.1549`
  - token delta norm p90 = `7.5515`
  - top-90-percentile changed-token ratio = `0.1000`
  - EEF proxy region 平均 delta norm 更高，但这只是 EEF proxy，不是真 object-hand mask。
- action-coupling diagnostic 使用 ridge regression：
  - current-only MSE = `0.00040323`
  - current+action MSE = `0.00033855`
  - relative gain = `0.1604`

这个结果支持一个更窄的判断：future delta 中确实有 action-coupled signal，值得训练 `delta_future` / `changed_token_future` 作为机制对照。

验证状态：

- `scripts/smoke_test_trajectory_future_modes.py` 已通过。
- 既有 `scripts/smoke_test_future_modes.py` 已通过。
- 新增 Python 文件和修改的 policy/helper 文件已通过 `py_compile`。

未启动：

- eval100 baseline 没有在本轮启动。
- `delta_future` / `changed_token_future` 训练没有在本轮启动。

原因是 100-rollout eval 和 200-epoch training 都会产生较大输出和较长运行时间；本轮先完成可复现实验入口、诊断和代码护栏。

## 5. 下一步实验

建议顺序：

1. 先运行 4 个 key baseline 的 eval100，确认 public-50 低成功率是否稳定：
   - `vanilla`
   - `dino_only`
   - `no_future`
   - `pi3_pooled`
2. 再训练并 eval：
   - `delta_future`
   - `changed_token_future`
   - 可选：`delta_changed_token_future`
3. 比较：
   - `vanilla`
   - `no_future`
   - `pi3_pooled`
   - `delta_future`
   - `changed_token_future`
4. 如果 delta/changed-token 有收益，再考虑 true segmentation/object-hand validation。
5. 暂时不做 100-demo 主复现，不换任务。

推荐下一条命令：

```bash
bash scripts/generated_dual_shoes_eval100_commands.sh
```

baseline eval100 稳定后再运行：

```bash
bash scripts/generated_dual_shoes_trajectory_commands.sh
```

## 6. 需要导师反馈的问题

- 是否认可把当前工作定义为 public-50 Place Dual Shoes mechanism study？
- 是否认可先做 delta/changed-token future supervision，而不是直接补 object masks？
- 如果 changed-token future 有收益，是否下一步再做 true segmentation/object-hand validation？
- 是否需要补 seed1/seed2，还是先把 seed0 机制跑通？
