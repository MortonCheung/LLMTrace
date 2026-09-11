"""Fingerprint executor —— 探测编排层（Task 8）.

边界（Rule 4 / Task 8）：

- 它**不是** Provider，不持有 HTTP client，也不自行 retry；
- 每个真实请求都经由 ``Provider.complete()``，因此 budget / redaction /
  evidence / secret safety 全部由 Provider 统一承担；
- 失败（http != 2xx / exception / 空输出 / 归一化失败）一律记为
  ``__INVALID__``，**绝不缩小 denominator**，也不 silent retry。

轮换顺序（Task 8.1）：按 rounds 交错执行，而不是"probe A × N 再 probe B × N"，
避免时间漂移与模型 routing 变化被误读成 probe 差异。
"""

from __future__ import annotations

from collections.abc import Sequence, Set

from llmtrace.benchmarks.models import CompletionOptions
from llmtrace.execution.progress import (
    EVENT_PROGRESS,
    STAGE_FINGERPRINTING,
    CancellationToken,
    ProgressEvent,
    ProgressSink,
)
from llmtrace.fingerprint.models import (
    INVALID_OUTCOME,
    FingerprintProbe,
    FingerprintSampleObservation,
    FingerprintSuite,
)
from llmtrace.fingerprint.normalizers import ChoiceNormalizer, resolve_normalizer
from llmtrace.fingerprint.suite import probe_order
from llmtrace.models.evidence import EvidenceType, HTTPEvidence
from llmtrace.providers.base import BaseProvider

#: 写入 HTTPEvidence.evidence_type 的取值（Task 7 / Task 27 closure validation）.
FINGERPRINT_EVIDENCE_TYPE: str = EvidenceType.FINGERPRINT_PROBE.value


class FingerprintEvidenceClosureError(Exception):
    """观测引用了本次 run 中并不存在的 evidence（Task 27）."""

    error_code = "FINGERPRINT_EVIDENCE_CLOSURE_ERROR"


def assert_evidence_closure(
    observations: Sequence[FingerprintSampleObservation],
    evidence_ids: Set[str],
) -> None:
    """每个观测的 ``evidence_ref`` 必须真实存在于本次 run 的证据集合中（Task 27）.

    「不生成虚构 refs」是证据链的最低要求：一旦某个 ref 不在 run 的
    ``HTTPEvidence.evidence_id`` 里，这份观测就无法被独立复核 —— fail closed，
    绝不把不可复核的观测写进 artifact。

    Raises:
        FingerprintEvidenceClosureError: 存在无法在本次 run 证据中定位的 ref。
    """
    referenced = {observation.evidence_ref for observation in observations}
    missing = sorted(referenced - evidence_ids)
    if missing:
        raise FingerprintEvidenceClosureError(
            f"{len(missing)} fingerprint observation(s) reference evidence that is not part of this run's "
            f"recorded evidence: {missing[:5]}"
        )


class FingerprintExecutor:
    """按轮次执行探测并把响应转成观测（不发送旁路请求）."""

    def __init__(
        self,
        *,
        provider: BaseProvider,
        suite: FingerprintSuite,
        progress_sink: ProgressSink | None = None,
        cancel_token: CancellationToken | None = None,
    ) -> None:
        self._provider = provider
        self._suite = suite
        self._sink = progress_sink
        self._cancel_token = cancel_token
        # 归一化策略在构造期解析一次；不受支持的策略在此 fail closed。
        self._normalizer: ChoiceNormalizer = resolve_normalizer(
            suite.normalization_policy_id,
            suite.normalization_policy_version,
        )

    @property
    def suite(self) -> FingerprintSuite:
        """本次采集使用的探测套件."""
        return self._suite

    async def run(
        self,
        *,
        model: str,
        repetitions: int,
        seed: int = 0,
    ) -> tuple[FingerprintSampleObservation, ...]:
        """执行 ``repetitions`` 轮探测，返回按真实请求顺序排列的观测.

        Raises:
            ValueError: repetitions < 1（在发出任何请求之前 fail fast）。
            RunCancelledError: 协作取消（状态映射为 CANCELLED）。
        """
        if repetitions < 1:
            raise ValueError(f"repetitions must be >= 1, got {repetitions}")

        probes = self._suite.probes
        total = len(probes) * repetitions
        observations: list[FingerprintSampleObservation] = []
        completed = 0

        for round_index in range(repetitions):
            ordered = probe_order(probes, seed=seed, round_index=round_index)
            for sequence_index, probe in enumerate(ordered):
                # 每个请求前检查取消：取消后不再发出后续请求（Task 25）。
                if self._cancel_token is not None:
                    self._cancel_token.throw_if_cancelled()

                evidence = await self._provider.complete(
                    model,
                    [{"role": "user", "content": probe.prompt}],
                    options=CompletionOptions(
                        temperature=probe.temperature,
                        max_tokens=probe.max_output_tokens,
                    ),
                    evidence_type=FINGERPRINT_EVIDENCE_TYPE,
                )

                observations.append(
                    self._build_observation(
                        probe,
                        evidence,
                        sequence_index=sequence_index,
                        round_index=round_index,
                    )
                )
                completed += 1
                self._emit_progress(completed, total)

        return tuple(observations)

    def _build_observation(
        self,
        probe: FingerprintProbe,
        evidence: HTTPEvidence,
        *,
        sequence_index: int,
        round_index: int,
    ) -> FingerprintSampleObservation:
        """把一次请求结果转成观测；任何失败都不缩小分母."""
        if evidence.success and evidence.response_text.strip():
            outcome, valid = self._normalizer(evidence.response_text, probe.choices)
        else:
            outcome, valid = INVALID_OUTCOME, False

        return FingerprintSampleObservation(
            probe_id=probe.probe_id,
            sequence_index=sequence_index,
            round_index=round_index,
            outcome=outcome,
            valid=valid,
            response_model=evidence.response_model,
            finish_reason=evidence.finish_reason,
            total_latency_ms=evidence.total_latency_ms,
            first_token_latency_ms=evidence.first_token_latency_ms,
            input_tokens=evidence.input_tokens,
            output_tokens=evidence.output_tokens,
            response_body_sha256=evidence.response_body_sha256,
            evidence_ref=str(evidence.evidence_id),
        )

    def _emit_progress(self, completed: int, total: int) -> None:
        """向 sink 通报进度（无 sink 时为 no-op，CLI 行为不变）."""
        if self._sink is None:
            return
        self._sink(
            ProgressEvent(
                type=EVENT_PROGRESS,
                stage=STAGE_FINGERPRINTING,
                message=f"fingerprint probe {completed}/{total}",
                completed=completed,
                total=total,
            )
        )


__all__: list[str] = [
    "FINGERPRINT_EVIDENCE_TYPE",
    "FingerprintEvidenceClosureError",
    "FingerprintExecutor",
    "assert_evidence_closure",
]
