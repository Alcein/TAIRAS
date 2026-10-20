"""Rewrite frozen TAIRA queries so the assigned interaction state is observable.

The legacy TAIRA queries were authored for single-turn recommendation.  A
quota-constrained allocator later assigned them to multi-turn mechanisms, but
some low-score assignments are not distinguishable from the public query.
This one-time authoring command preserves the legacy shopping scenario while
making only the *unresolved decision structure* visible.  It never supplies
the frozen user answer to the rewriting model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from taira_mt.multiturn_benchmark.runtime import load_frozen_agent_records
from taira_mt.provenance import canonical_json, content_sha256
from taira_mt.providers.openai_compatible import (
    OpenAICompatibleTransportProfile,
    build_openai_compatible_json_transport,
)

STATE_OBSERVABLE_QUERY_VERSION = "state_observable_initial_query_v1"
REWRITE_MECHANISMS = frozenset({"progressive", "ambiguous", "conflict", "revision"})

_PROMPT = """You rewrite shopping requests for a multi-turn recommendation benchmark.
Preserve the original product, scenario, audience, constraints, and tone. Make the unresolved decision structure
observable without revealing which option the user will ultimately choose.

For progressive cases, naturally show that one supplied decision axis remains open.
For ambiguous cases, naturally expose the two plausible interpretations in the supplied clarification question.
For conflict cases, state that both supplied goals matter but do not choose which should dominate.
For revision cases, preserve a coherent initial need and naturally state that the shopper prefers to react to one
concrete initial option before refining details. Do not reveal the later adjustment in the supplied question.

The rewritten request must remain a natural shopper utterance of one or two sentences. Do not mention a benchmark,
mechanism, hidden state, agent, expected action, or answer. Do not answer the supplied clarification question.
Return exactly one JSON object with key rewrites. rewrites must contain one object per input with exactly
episode_id and rewritten_query."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create state-observable public queries while preserving legacy TAIRA scenarios."
    )
    parser.add_argument(
        "--source-artifact",
        type=Path,
        default=Path("artifacts/frozen_multiturn_test_v1/agent_test_episodes.jsonl"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/frozen_multiturn_test_v3_state_observable"),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider-profile-id", required=True)
    parser.add_argument("--api-key-env", default="TAIRA_MT_API_KEY")
    parser.add_argument("--base-url-env", default="TAIRA_MT_BASE_URL")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--rewrite-mechanisms",
        default="progressive,ambiguous,conflict,revision",
        help="Comma-separated subset of progressive, ambiguous, conflict, revision.",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-completion-tokens", type=int, default=3200)
    parser.add_argument("--disable-thinking", action="store_true")
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


def _parse_rewrites(content: str, expected_ids: list[str]) -> dict[str, str]:
    value = json.loads(content)
    if not isinstance(value, dict) or set(value) != {"rewrites"}:
        raise ValueError("rewrite response must contain exactly the rewrites key")
    rows = value["rewrites"]
    if not isinstance(rows, list):
        raise ValueError("rewrites must be a list")
    output: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"episode_id", "rewritten_query"}:
            raise ValueError("every rewrite must contain exactly episode_id and rewritten_query")
        episode_id = _clean(row["episode_id"])
        query = _clean(row["rewritten_query"])
        if episode_id in output:
            raise ValueError(f"duplicate rewrite: {episode_id}")
        if len(query) < 20 or len(query) > 700:
            raise ValueError(f"invalid rewritten query length: {episode_id}")
        forbidden = ("benchmark", "hidden state", "expected action", "the agent should")
        if any(marker in query.casefold() for marker in forbidden):
            raise ValueError(f"rewrite leaks construction language: {episode_id}")
        output[episode_id] = query
    if set(output) != set(expected_ids):
        raise ValueError("rewrite response episode IDs do not match the requested batch")
    return output


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    output: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            output[str(row["batch_id"])] = row
    return output


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.batch_size > 20:
        raise SystemExit("batch-size must be between 1 and 20.")
    if args.max_attempts < 1:
        raise SystemExit("max-attempts must be positive.")
    requested_mechanisms = {
        value.strip() for value in args.rewrite_mechanisms.split(",") if value.strip()
    }
    if not requested_mechanisms or not requested_mechanisms <= REWRITE_MECHANISMS:
        raise SystemExit("rewrite-mechanisms contains unsupported mechanisms.")
    if args.output_root.exists() and not (args.output_root / "rewrite_cache.jsonl").is_file():
        raise FileExistsError(f"Output directory already exists and cannot be resumed: {args.output_root}")

    print("Step 1/4: reading the frozen benchmark and selecting non-Direct queries that need observable state information.", flush=True)
    records = load_frozen_agent_records(args.source_artifact)
    rewrite_records = [
        record for record in records if _clean(record.get("mechanism")) in requested_mechanisms
    ]
    print(
        f"Stage complete: {len(records)} records in total, {len(rewrite_records)} require a one-time rewrite; "
        "Direct queries retain the original TAIRA wording.",
        flush=True,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_root / "rewrite_cache.jsonl"
    cache = _load_cache(cache_path)
    profile = OpenAICompatibleTransportProfile(
        provider_profile_id=args.provider_profile_id,
        model=args.model,
        api_key_env=args.api_key_env,
        base_url_env=args.base_url_env,
        temperature=0.0,
        max_completion_tokens=args.max_completion_tokens,
        enable_thinking=False if args.disable_thinking else None,
    )
    transport = build_openai_compatible_json_transport(profile)

    print("Step 2/4: rewriting public queries in batches; the model sees only the original query and clarification questions, not final user answers.", flush=True)
    rewritten: dict[str, str] = {}
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "api_batches": 0}
    for start in range(0, len(rewrite_records), args.batch_size):
        batch = rewrite_records[start : start + args.batch_size]
        inputs = [
            {
                "episode_id": _clean(record["episode_id"]),
                "mechanism": _clean(record["mechanism"]),
                "original_query": _clean(record["initial_query"]),
                "unresolved_decision_question": _clean(
                    ""
                    if record["mechanism"] == "revision"
                    else record.get("transition", {}).get("agent_action_required")
                ),
                "visible_interaction_preference": (
                    "The shopper wants to see one concrete initial option before refining details."
                    if record["mechanism"] == "revision"
                    else ""
                ),
            }
            for record in batch
        ]
        batch_id = content_sha256(inputs)
        cached = cache.get(batch_id)
        if cached is not None:
            batch_rewrites = {
                str(key): str(value) for key, value in cached["rewrites"].items()
            }
        else:
            last_error: Exception | None = None
            for attempt in range(1, args.max_attempts + 1):
                try:
                    completion = transport.complete_json(
                        system_prompt=_PROMPT,
                        user_payload={"cases": inputs},
                        retry_feedback=(
                            None
                            if attempt == 1
                            else "Return one valid rewrite for every supplied episode ID and no extra keys."
                        ),
                    )
                    batch_rewrites = _parse_rewrites(
                        completion.content,
                        [_clean(record["episode_id"]) for record in batch],
                    )
                    usage["input_tokens"] += completion.usage.input_tokens
                    usage["output_tokens"] += completion.usage.output_tokens
                    usage["total_tokens"] += completion.usage.total_tokens
                    usage["api_batches"] += 1
                    cache_row = {
                        "batch_id": batch_id,
                        "rewrites": batch_rewrites,
                        "requested_model": completion.requested_model,
                        "returned_model": completion.returned_model,
                        "usage": completion.usage.model_dump(mode="json"),
                    }
                    with cache_path.open("a", encoding="utf-8") as handle:
                        handle.write(canonical_json(cache_row) + "\n")
                    cache[batch_id] = cache_row
                    break
                except Exception as exc:
                    last_error = exc
            else:
                assert last_error is not None
                raise last_error
        rewritten.update(batch_rewrites)
        completed = min(start + len(batch), len(rewrite_records))
        print(f"Progress: rewrote {completed}/{len(rewrite_records)} queries.", flush=True)

    print("Step 3/4: freezing the new public surface and removing pseudo-relevance sets coupled to the lexical retriever.", flush=True)
    output_records: list[dict[str, Any]] = []
    changed = Counter()
    for source in records:
        record = json.loads(json.dumps(source, ensure_ascii=False))
        episode_id = _clean(record["episode_id"])
        mechanism = _clean(record["mechanism"])
        original_query = _clean(record["initial_query"])
        if episode_id in rewritten:
            record["initial_query"] = rewritten[episode_id]
            changed[mechanism] += 1
        contract = record.get("evaluation_contract", {})
        if isinstance(contract, dict):
            contract.pop("resolved_need_relevance_item_ids", None)
            contract.pop("resolved_need_relevance_provenance", None)
        if episode_id in rewritten:
            record["query_surface_provenance"] = {
                "version": STATE_OBSERVABLE_QUERY_VERSION,
                "original_query_sha256": content_sha256(original_query),
                "rewritten": True,
                "rewritten_mechanism": mechanism,
                "author_saw_frozen_user_answer": False,
                "author_saw_later_revision_detail": False if mechanism == "revision" else None,
                "preserves_legacy_scenario": True,
                "parent_provenance": source.get("query_surface_provenance"),
            }
        output_records.append(record)

    artifact_path = args.output_root / "agent_test_episodes.jsonl"
    artifact_path.write_text(
        "".join(canonical_json(record) + "\n" for record in output_records),
        encoding="utf-8",
    )
    summary = {
        "version": STATE_OBSERVABLE_QUERY_VERSION,
        "source_artifact": str(args.source_artifact),
        "source_sha256": _sha256_file(args.source_artifact),
        "output_sha256": _sha256_file(artifact_path),
        "record_count": len(output_records),
        "rewritten_count": len(rewritten),
        "rewritten_by_mechanism": dict(sorted(changed.items())),
        "requested_rewrite_mechanisms": sorted(requested_mechanisms),
        "unchanged_mechanisms": ["direct"],
        "removed_relevance_contract": "resolved_need_pseudo_relevance_v1",
        "provider_profile": profile.secret_free_telemetry(),
        "new_api_usage": usage,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Stage complete: new benchmark surface written to {artifact_path.resolve()}.", flush=True)

    print("Step 4/4: reporting frozen statistics.", flush=True)
    print(
        f"Complete: rewrote {len(rewritten)} records, retained {len(output_records) - len(rewritten)}; "
        f"new calls in this run: {usage['api_batches']} batches, {usage['total_tokens']} tokens.",
        flush=True,
    )


if __name__ == "__main__":
    main()
