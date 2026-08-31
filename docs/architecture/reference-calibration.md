# Trusted Reference Run & Reference Set Foundation

v0.4-A 为 v0.4-B Calibration 建立可信、不可变、可复现、可审计的参考事实基础设施：
一条 Operator 确认可信的运行（**Reference Run**）→ 通过 10 道资格门禁（**Reference
Qualification**）→ 生成带完整 provenance 的 **ReferenceSnapshot** → 多个已验证快照组成
**ReferenceSet**。领域模型见 `src/llmtrace/reference/`。

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

## 语义边界

```text
ReferenceSnapshot   = 一次已验证参考运行的不可变能力事实
ReferenceSet        = 多个兼容快照的版本化组合
CalibrationPolicy   = future（v0.4-B）从 ReferenceSet 推导 0–100 的映射规则

ReferenceSnapshot != ReferenceSet != CalibrationPolicy
Reference Comparison != Calibration
ReferenceSet 当前不产生 calibrated 0–100 分数
```

## CLI

```bash
llmtrace reference capture    # Operator 对可信 endpoint 捕获参考运行（--dry-run 0 副作用）
llmtrace reference set-create # 从已验证快照构建 ReferenceSet（0 API 请求）
```

## 参考文档

- `docs/roadmap.md` v0.4-A / v0.4-B
- `docs/product/PRD.md` 当前开发基线
- `docs/architecture/unified-execution.md` 统一执行链（capture 复用 `UnifiedAuditRunner`）
