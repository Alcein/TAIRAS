"""Build persistent full-catalog retrieval indexes for the paper Agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from taira_mt.data.legacy_catalog import LegacyCatalogAdapter
from taira_mt.mt_agent.retrieval import LexicalCatalogRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build shared TAIRAS full-catalog lexical indexes.")
    parser.add_argument("--data-root", type=Path, default=Path("data/catalogs"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/mt_catalog_indexes_v1"))
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["amazon_clothing", "amazon_beauty", "amazon_music"],
    )
    parser.add_argument("--max-features", type=int, default=120_000)
    parser.add_argument("--min-df", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("步骤 1/3：检查完整 Amazon 商品库并准备共享检索索引。", flush=True)
    if args.max_features < 1 or args.min_df < 1:
        raise SystemExit("max-features 和 min-df 必须为正数。")
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"阶段完成：将为 {len(args.domains)} 个域构建独立、可缓存的公开检索器。", flush=True)
    for position, domain in enumerate(args.domains, start=1):
        metadata_path = args.data_root / domain / "metadata.csv"
        print(f"步骤 2/3：正在建立 {domain} 的全库词法索引（{position}/{len(args.domains)}）。", flush=True)
        adapter = LegacyCatalogAdapter(metadata_path, domain=domain)
        index = adapter.load_index()
        pool = index.pool(
            # Keep the exact pool identity used by FrozenMultiTurnBenchmark.
            # The retriever is shared infrastructure, so its rankings must be
            # attachable to a live benchmark session without a hidden pool swap.
            recipe_id="full_legacy_amazon_catalog_v1",
            recipe_config={"domain": domain, "scope": "all_metadata_items"},
        )
        retriever = LexicalCatalogRetriever.build(
            pool,
            max_features=args.max_features,
            min_df=args.min_df,
        )
        output_path = args.output_root / domain / "lexical_index.joblib"
        retriever.save(output_path)
        print(
            f"阶段完成：{domain} 已索引 {len(retriever.item_ids)} 件商品，写入 {output_path.resolve()}。",
            flush=True,
        )
    print("步骤 3/3：全部共享检索索引已完成；后续所有 Agent 和 baseline 将使用同一检索面。", flush=True)


if __name__ == "__main__":
    main()
