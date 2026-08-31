"""BehaviorSnapshotBuilder — assemble BehaviorRunSnapshot from run artefacts."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from llmtrace.benchmarks.models import BenchmarkRunResult
from llmtrace.models.evidence import HTTPEvidence
from llmtrace.scoring.models import CapabilityProfile

from .behavior_models import (
    BehaviorItemKey,
    BehaviorItemObservation,
    BehaviorRunSnapshot,
    DuplicateEvidenceError,
    DuplicateItemKeyError,
    ItemEvidenceError,
    MissingItemIdentityError,
    generation_config_sha256,
    output_text_sha256,
)


class BehaviorSnapshotBuilder:
    """Build an immutable ``BehaviorRunSnapshot`` from benchmark run artefacts.

    Contract:

    - Every drift item must have ``source_sample_id`` and ``input_sha256``.
    - Stable ``BehaviorItemKey`` duplicates are rejected.
    - Every item must resolve to **exactly one** HTTP evidence.  Zero or more
      than one reference fails closed — the builder never guesses a primary.
    - Item ordering is deterministic (sorted by stable key).
    - The full model output is never stored; only the canonicalized hash and
      length are captured.
    """

    def build(
        self,
        *,
        run_results: Sequence[BenchmarkRunResult],
        profile: CapabilityProfile,
        evidence: Sequence[HTTPEvidence],
        target_id: str,
        candidate_model_id: str,
        generation_config: Any,
        created_at: datetime | None = None,
    ) -> BehaviorRunSnapshot:
        """Build a snapshot.

        Args:
            run_results: The benchmark run result(s) that produced *profile*.
                A Quick Suite run is one result per task; all must share the
                same suite and adapter.
            profile: The aggregated ``CapabilityProfile`` for the run.
            evidence: All HTTP evidence collected during the run.
            target_id: Stable caller-supplied label for the target API.
            candidate_model_id: Candidate model label.
            generation_config: Generation configuration (Mapping or model).
            created_at: Snapshot timestamp (defaults to the latest run finish).

        Raises:
            MissingItemIdentityError: If an item lacks source_sample_id or input_sha256.
            DuplicateItemKeyError: If two items share a stable key.
            ItemEvidenceError: If an item's evidence is missing or ambiguous.
            ValueError: If run_results are empty or disagree on suite/adapter.
        """
        if not run_results:
            raise ValueError("run_results must not be empty")

        suite_id, suite_version, adapter_id, adapter_version = self._uniform_context(run_results)

        evidence_map = self._build_evidence_map(evidence)

        source_pairs = sorted({(rr.source_id, rr.source_revision) for rr in run_results})
        source_ids = tuple(pair[0] for pair in source_pairs)
        source_revisions = tuple(pair[1] for pair in source_pairs)

        observations: list[BehaviorItemObservation] = []
        seen_keys: set[BehaviorItemKey] = set()

        for run in run_results:
            for attempt in run.task_attempts:
                for item in attempt.item_results:
                    if item.source_sample_id is None or item.input_sha256 is None:
                        raise MissingItemIdentityError(
                            f"Item '{item.item_id}' (task '{item.task_id}') is missing "
                            f"source_sample_id or input_sha256; cannot build a stable drift identity"
                        )
                    key = BehaviorItemKey(
                        task_id=item.task_id,
                        source_sample_id=item.source_sample_id,
                        input_sha256=item.input_sha256,
                    )
                    if key in seen_keys:
                        raise DuplicateItemKeyError(
                            f"Duplicate stable item key '{key.key_string()}' in run '{run.run_id}'"
                        )
                    seen_keys.add(key)

                    observations.append(
                        self._build_observation(
                            item.status, item.raw_score, item.normalized_score, key, item.evidence_refs, evidence_map
                        )
                    )

        observations.sort(key=lambda o: o.key.sort_key)

        union_refs: list[str] = []
        seen_refs: set[str] = set()
        for obs in observations:
            for ref in obs.evidence_refs:
                if ref not in seen_refs:
                    seen_refs.add(ref)
                    union_refs.append(ref)

        snapshot_created_at = created_at if created_at is not None else self._derive_created_at(run_results)

        return BehaviorRunSnapshot(
            run_id=self._derive_run_id(run_results),
            target_id=target_id,
            candidate_model_id=candidate_model_id,
            created_at=snapshot_created_at,
            suite_id=suite_id,
            suite_version=suite_version,
            source_ids=source_ids,
            source_revisions=source_revisions,
            adapter_id=adapter_id,
            adapter_version=adapter_version,
            scoring_policy_id=profile.scoring_policy_id,
            scoring_policy_version=profile.scoring_policy_version,
            generation_config_sha256=generation_config_sha256(generation_config),
            capability_profile=profile,
            items=tuple(observations),
            evidence_refs=tuple(union_refs),
        )

    # -- Helpers -----------------------------------------------------------

    @staticmethod
    def _uniform_context(
        run_results: Sequence[BenchmarkRunResult],
    ) -> tuple[str, str, str, str]:
        """Return the (suite_id, suite_version, adapter_id, adapter_version) shared by all runs."""
        first = run_results[0]
        context = (first.suite_id, first.suite_version, first.adapter_id, first.adapter_version)
        for run in run_results[1:]:
            other = (run.suite_id, run.suite_version, run.adapter_id, run.adapter_version)
            if other != context:
                raise ValueError(f"run_results disagree on suite/adapter context: {context} vs {other}")
        return context

    @staticmethod
    def _build_evidence_map(evidence: Sequence[HTTPEvidence]) -> dict[str, HTTPEvidence]:
        """Build an evidence_id → HTTPEvidence map, rejecting duplicate ids.

        A duplicate evidence_id would otherwise silently last-write-wins, which
        is exactly the kind of ambiguity a snapshot must not paper over.
        """
        evidence_map: dict[str, HTTPEvidence] = {}
        for ev in evidence:
            eid = str(ev.evidence_id)
            if eid in evidence_map:
                raise DuplicateEvidenceError(f"duplicate evidence_id '{eid}'; each evidence must have a unique id")
            evidence_map[eid] = ev
        return evidence_map

    @staticmethod
    def _derive_created_at(run_results: Sequence[BenchmarkRunResult]) -> datetime:
        finished = [r.finished_at for r in run_results if r.finished_at is not None]
        if finished:
            latest = max(finished)
            if latest.tzinfo is None or latest.tzinfo.utcoffset(latest) is None:
                raise ValueError(
                    f"finished_at must be timezone-aware, got naive datetime {latest.isoformat()}; "
                    f"a historical instant without a timezone cannot be silently interpreted as UTC"
                )
            return latest.astimezone(UTC)
        return datetime.now(UTC)

    @staticmethod
    def _derive_run_id(run_results: Sequence[BenchmarkRunResult]) -> str:
        run_ids = sorted({r.run_id for r in run_results})
        if len(run_ids) == 1:
            return run_ids[0]
        return "::".join(run_ids)

    @staticmethod
    def _build_observation(
        status: Any,
        raw_score: float,
        normalized_score: float,
        key: BehaviorItemKey,
        item_evidence_refs: Sequence[str],
        evidence_map: dict[str, HTTPEvidence],
    ) -> BehaviorItemObservation:
        """Resolve the single primary evidence and construct an observation.

        Fails closed when the item references zero or more than one evidence.
        """
        if len(item_evidence_refs) != 1:
            raise ItemEvidenceError(
                f"Item '{key.key_string()}' references {len(item_evidence_refs)} evidence refs; "
                f"exactly one primary HTTP evidence is required"
            )
        ref = item_evidence_refs[0]
        ev = evidence_map.get(ref)
        if ev is None:
            raise ItemEvidenceError(f"Item '{key.key_string()}' references unknown evidence '{ref}'")

        text = ev.response_text
        return BehaviorItemObservation(
            key=key,
            status=status,
            raw_score=raw_score,
            normalized_score=normalized_score,
            output_text_sha256=output_text_sha256(text),
            output_length=len(text),
            response_body_sha256=ev.response_body_sha256 or "",
            response_model=ev.response_model,
            finish_reason=ev.finish_reason,
            latency_ms=ev.total_latency_ms,
            input_tokens=ev.input_tokens,
            output_tokens=ev.output_tokens,
            evidence_refs=tuple(item_evidence_refs),
        )
