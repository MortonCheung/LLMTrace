# Trusted Reference Run & Reference Set / Reference Calibration

v0.4-A 为 v0.4-B Calibration 建立可信、不可变、可复现、可审计的参考事实基础设施：
一条 Operator 确认可信的运行（**Reference Run**）→ 通过 10 道资格门禁（**Reference
Qualification**）→ 生成带完整 provenance 的 **ReferenceSnapshot** → 多个已验证快照组成
**ReferenceSet**。领域模型见 `src/llmtrace/reference/`。

v0.4-B 在此之上实现 **Reference Anchored Monotonic Piecewise Calibration**：从可信
ReferenceSet 推导正式 0–100 能力分（**Capability Score**），并输出被测 endpoint 与
**声明模型**可信参考配置之间的能力差距（**Claimed Model Gap**）。校准领域模型见
`src/llmtrace/scoring/calibration.py`。

本文件记录这层架构约定与语义边界。

## 一次参考运行的完整生命周期

```text
Operator 确认 endpoint 为可信参考源
→ llmtrace reference capture
→ UnifiedAuditRunner 执行统一运行（协议 + Quick Suite 32 题 + 能力画像）
→ RunArtifactRepository 落盘（manifest + benchmark_runs.json + capability_profile.json）
→ Reference Qualification（Gate 1–10，fail closed）
→ ReferenceSnapshotBuilder（verify → qualify → build → save）
→ ReferenceSetBuilder（12-gate Compatibility）+ ReferenceSetRepository（append-only）
```

每个概念对应一个不可混淆的产物：

- **Reference Run** = 一次落盘的运行工件（`RunArtifactRepository` 管理的不可变目录）。
  即使资格被拒，工件仍保留——失败/被拒的测量同样是历史事实。
- **Reference Qualification** = 对落盘工件执行的 10 道门禁链，任何一道失败即 REJECTED
  （fail closed），不产生快照，并返回机器可读 `reason_codes`。
- **ReferenceSnapshot** = 一次已验证参考运行的不可变能力事实（`snapshot_id` 唯一，append-only）。
- **ReferenceSet** = 一组互相兼容的已验证快照（同 suite / adapter / scoring policy /
  generation config / qualification policy / 维度覆盖），带 canonical content hash 自校验。

## Reference Qualification（Gate 1–10）

门禁按序 fail closed，只有全部通过才 `QUALIFIED`：

1. **Artifact Integrity**：`RunArtifactRepository.verify(execution_id)`——manifest 与 artifacts
   SHA 一致、未篡改。
2. **Capability Profile 存在**：manifest.artifacts 必须记录 `capability_profile.json`。
3. **读落盘已验证 profile**：资格判定只使用磁盘上通过验证的 profile，绝不使用运行期的瞬时对象。
4. **Measurement 完整**：32/32 全部 GRADED、0 failure、0 ungradable。
5. **Scoring Policy 全链一致**：plan / profile / manifest 的 scoring policy id+version 一致。
6. **Suite**：suite_id / suite_version 正确且 `suite_content_sha256` 与 Quick Suite manifest 的
   canonical content hash 一致（缺失 content SHA 的旧运行永不自动成为参考）。
7. **Generation Config**：manifest 的 `generation_config_sha256` 与 canonical 配置一致。
8. **Adapter**：manifest 必须记录非空 `adapter_id` / `adapter_version`。
9. **Capability Coverage**：profile 的 `coverage_weight` 等于 scoring policy 计算值（当前 0.75）。
10. **Dimension Coverage**：profile 覆盖全部 enabled 维度，且无非可测量状态
    （UNAVAILABLE / INSUFFICIENT_DATA）。

## Suite Content Identity

`suite_content_sha256` 是 Quick Suite 的 canonical 内容身份：

- 对 manifest 构造**语义 payload**（suite_id / suite_version / total_items / selection_algorithm /
  selection_seed_format + 确定性排序的 tasks[]，每项含 task_id / dimension / source_id /
  upstream_revision / subset_sha256 / sample_count / adapter_id）。
- `json.dumps(sort_keys=True, separators=(",",":"), ensure_ascii=True)` → UTF-8 → SHA-256。
- **绝不直接 hash manifest.json 原始字节**：JSON key 顺序或空白改变不改变内容身份。
- `get_quick_suite_source_revisions()` 以 manifest 为单一真相源返回 `task_id → upstream_revision`，
  不二次硬编码 revision 表。

## ReferenceSet 一致性

`ReferenceSetBuilder` 先对每个成员做 12-gate Compatibility 检查（suite / suite_version /
suite_content_sha256 / adapter_id / adapter_version / scoring_policy_id / scoring_policy_version /
generation_config_sha256 / qualification_policy_id / qualification_policy_version / 可比维度集合 /
coverage_weight），任何不一致 fail closed。同一模型允许在同一 set 中保留多个快照（不同时点）。

安全约束：

- 生产 builder 拒绝 `source_type == "test_fixture"`（测试夹具永不是可信参考事实；
  测试可显式 `allow_test_fixture=True`）。
- API Key 仅存在于内存；endpoint 只存 `endpoint_redacted`（凭据已脱敏）。
- Reference 层只存 hash / provenance / profile / 不可变指针，不存 secret。

## v0.4-B Reference Calibration（正式 0–100 能力分）

### 校准链路

```text
ReferenceSet JSON（--reference-set）
→ Preflight：validate_reference_set_for_calibration（信任链重验：sidecar / SHA / identity / profile SHA / run manifest / source_type）
→ aggregate_reference_identities（同 model 跨 provider 聚合，重复快照取中位数）
→ build_calibration_curves（每维度 + 总分锚点曲线）
→ calibrate_capability_profile（raw → 0–100）
→ compute_claimed_model_gap（声明模型差距）
→ Console / JSON / HTML 报告 + manifest 校准 provenance
```

### CalibrationPolicy v1（`llmtrace-reference-calibration-v1` / 0.1.0）

版本化、不可变的映射规则；改变规则必须开新 policy 版本，不能只换 id。v1 锁定：

- **分段线性（piecewise linear）单调映射**，锚点：`0 = random floor`（各维度随机基线，
  总分为加权 floor）、`50 = 参考组中位数`、`90 = P90（flagship_quantile=0.90）`、
  `100 = 套件上限`（维度 raw=1.0；总分为 coverage_weight=0.75）。
- **最少 5 个不同参考身份**（distinct reference identities）才允许校准。
- 超出锚点区间的 raw 值 clamp 到 0–100。
- 同一模型多次快照取中位数（同模型多时点快照不重复占据身份）。

### Fail-Closed 条件（不产生任何假分数）

任一条件触发即跳过校准（warning + 保持 `UNCALIBRATED`，原始分数与全部工件不受影响）：

- `< 5` 个不同参考身份，或任一维度参考数据不足；
- 任一维度 / 总分离散度不足（不同 raw 值 `< 3`，或中位数 ≤ random floor）；
- 校准饱和（P90 ≥ 套件上限）;
- 候选测量不完整（存在 FAILURE / UNGRADABLE 题目——正式分数不允许建立在部分测量上）;
- 候选 profile 的 scoring policy 与校准上下文不一致；
- ReferenceSet 信任链验证失败（preflight 直接拒绝，0 次 HTTP 请求）。

### Claimed Model Gap（声明模型差距）

- 声明模型 ID 与参考身份做**严格匹配**（不做模糊匹配 / 前缀匹配）。
- 恰好一个匹配 → 输出总分差距与分维度差距（candidate − reference，负值 = 低于参考）。
- 零个匹配 → gap 不可用（warning），不猜测最近邻。
- 多个匹配（同 model_id 出现在多个 provider）→ 拒绝任选其一（`AmbiguousClaimedModelError`）。
- **语义红线：这是能力对比，不是模型身份证明。** 报告中显式携带 interpretation 字段。

### 语义边界（v0.4-B 更新）

```text
ReferenceSnapshot   = 一次已验证参考运行的不可变能力事实
ReferenceSet        = 多个兼容快照的版本化组合（校准宇宙 calibration universe）
CalibrationPolicy   = 从 ReferenceSet 推导 0–100 的版本化映射规则
Capability Score    = 相对某个已定义校准宇宙的相对分数，不是绝对能力

ReferenceSnapshot != ReferenceSet != CalibrationPolicy
Reference Comparison != Calibration（前者是两个 profile 的原始对比，后者是 raw → 0–100 的正式映射）
Calibration Score is versioned：policy id+version + reference set id+version+content SHA 贯穿 provenance
0–100 is relative to a defined calibration universe：换参考组 = 换坐标系，分数不可跨组比较
Raw scores are retained：raw_normalized_score / provisional_raw_index 永远保留，校准只增不改
```

## CLI

```bash
llmtrace reference capture    # Operator 对可信 endpoint 捕获参考运行（--dry-run 0 副作用）
llmtrace reference set-create # 从已验证快照构建 ReferenceSet（0 API 请求）
llmtrace run --reference-set references/sets/refset-v1_0.1.0.json
# ↑ 统一审计 + 正式 0–100 校准 + 声明模型差距（preflight 即验证信任链）
```

## 参考文档

- `docs/roadmap.md` v0.4-A / v0.4-B
- `docs/product/PRD.md` 当前开发基线
- `docs/architecture/unified-execution.md` 统一执行链（capture 复用 `UnifiedAuditRunner`）
