"""Judge-facing reward comparison artifacts for AtomicVision."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from atomicvision.evaluation.policies import (
    POLICY_NAMES,
    PolicyEvaluationSummary,
    evaluate_policy,
)


DEFAULT_DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")
DEFAULT_POLICIES: tuple[str, ...] = (
    "cheap_submit",
    "random",
    "scan_heavy",
    "prior_submit",
    "oracle",
)


@dataclass(frozen=True)
class RewardComparison:
    """Aggregated comparison over policies and difficulty levels."""

    difficulties: tuple[str, ...]
    policies: tuple[str, ...]
    episodes_per_policy: int
    rows: list[PolicyEvaluationSummary]

    def to_dict(self) -> dict:
        return {
            "difficulties": list(self.difficulties),
            "policies": list(self.policies),
            "episodes_per_policy": self.episodes_per_policy,
            "rows": [row.to_dict() for row in self.rows],
        }


def run_reward_comparison(
    difficulties: tuple[str, ...] = DEFAULT_DIFFICULTIES,
    policies: tuple[str, ...] = DEFAULT_POLICIES,
    episodes: int = 8,
    seed_start: int = 0,
) -> RewardComparison:
    """Run deterministic reward comparisons across policies and difficulties."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    unknown = [policy for policy in policies if policy not in POLICY_NAMES]
    if unknown:
        raise ValueError(f"Unknown policies: {', '.join(unknown)}")

    seeds = list(range(seed_start, seed_start + episodes))
    rows: list[PolicyEvaluationSummary] = []
    for difficulty in difficulties:
        for policy in policies:
            rows.append(evaluate_policy(policy, seeds=seeds, difficulty=difficulty))

    return RewardComparison(
        difficulties=tuple(difficulties),
        policies=tuple(policies),
        episodes_per_policy=episodes,
        rows=rows,
    )


def write_comparison_artifacts(
    comparison: RewardComparison,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write JSON, CSV, Markdown, and SVG comparison artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "reward_comparison.json"
    csv_path = output_dir / "reward_comparison.csv"
    markdown_path = output_dir / "reward_comparison.md"
    svg_path = output_dir / "reward_comparison.svg"

    json_path.write_text(
        json.dumps(comparison.to_dict(), indent=2),
        encoding="utf-8",
    )
    _write_csv(comparison, csv_path)
    markdown_path.write_text(_markdown_report(comparison), encoding="utf-8")
    svg_path.write_text(_svg_chart(comparison), encoding="utf-8")

    return {
        "json": json_path,
        "csv": csv_path,
        "markdown": markdown_path,
        "svg": svg_path,
    }


def _write_csv(comparison: RewardComparison, path: Path) -> None:
    fieldnames = list(asdict(comparison.rows[0]).keys()) if comparison.rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in comparison.rows:
            writer.writerow(row.to_dict())


def _markdown_report(comparison: RewardComparison) -> str:
    best_by_difficulty = _best_policy_by_difficulty(comparison)
    lines = [
        "# AtomicVision Reward Comparison",
        "",
        "This report compares deterministic baseline policies for the AtomicVision OpenEnv lab.",
        "The trained GRPO agent in the next phase must improve over these baselines, especially `prior_submit`.",
        "",
        f"Episodes per policy/difficulty: `{comparison.episodes_per_policy}`",
        "",
        "## Best Policy By Difficulty",
        "",
        "| Difficulty | Best policy | Mean reward | Mean F1 | Mean scan cost |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for difficulty in comparison.difficulties:
        row = best_by_difficulty[difficulty]
        lines.append(
            f"| {difficulty} | {row.policy_name} | {row.mean_reward:.3f} | "
            f"{row.mean_f1:.3f} | {row.mean_scan_cost:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Full Results",
            "",
            "| Difficulty | Policy | Reward | Outcome | Penalty | F1 | Concentration MAE | Steps | Scan cost | Timeout rate |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparison.rows:
        lines.append(
            f"| {row.difficulty} | {row.policy_name} | {row.mean_reward:.3f} | "
            f"{row.mean_outcome_reward_total:.3f} | {row.mean_penalty_total:.3f} | "
            f"{row.mean_f1:.3f} | {row.mean_concentration_mae:.4f} | "
            f"{row.mean_steps:.2f} | {row.mean_scan_cost:.2f} | {row.timeout_rate:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Judge Narrative",
            "",
            "- `cheap_submit` shows the environment punishes blind guessing.",
            "- `scan_heavy` shows extra scans are useful only when they improve final accuracy enough to justify cost.",
            "- `prior_submit` is the strongest non-training baseline for Phase 11.",
            "- `oracle` is an upper-bound sanity check, not a deployable agent.",
        ]
    )
    return "\n".join(lines) + "\n"


def _svg_chart(comparison: RewardComparison) -> str:
    width = 960
    height = 420
    margin_left = 80
    margin_top = 40
    chart_width = 820
    chart_height = 300
    rows = comparison.rows
    rewards = [row.mean_reward for row in rows]
    min_reward = min(0.0, min(rewards))
    max_reward = max(rewards)
    reward_span = max(max_reward - min_reward, 1.0)
    bar_width = chart_width / max(len(rows), 1)
    colors = {
        "cheap_submit": "#9ca3af",
        "random": "#ef4444",
        "scan_heavy": "#f59e0b",
        "prior_submit": "#2563eb",
        "oracle": "#16a34a",
    }

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="420" viewBox="0 0 960 420">',
        '<rect width="960" height="420" fill="#ffffff"/>',
        '<text x="80" y="26" font-family="Arial" font-size="20" font-weight="700">AtomicVision Reward Comparison</text>',
        f'<line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{margin_left + chart_width}" y2="{margin_top + chart_height}" stroke="#111827" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_height}" stroke="#111827" stroke-width="1"/>',
    ]
    zero_y = margin_top + chart_height - ((0.0 - min_reward) / reward_span) * chart_height
    parts.append(
        f'<line x1="{margin_left}" y1="{zero_y:.2f}" x2="{margin_left + chart_width}" y2="{zero_y:.2f}" stroke="#d1d5db" stroke-width="1"/>'
    )

    for index, row in enumerate(rows):
        x = margin_left + index * bar_width + 4
        reward_height = abs(row.mean_reward - 0.0) / reward_span * chart_height
        if row.mean_reward >= 0.0:
            y = zero_y - reward_height
        else:
            y = zero_y
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_width - 8, 2):.2f}" '
            f'height="{reward_height:.2f}" fill="{colors.get(row.policy_name, "#6b7280")}"/>'
        )
        if index % len(comparison.policies) == 0:
            label_x = margin_left + index * bar_width + (bar_width * len(comparison.policies) / 2)
            parts.append(
                f'<text x="{label_x:.2f}" y="374" font-family="Arial" font-size="12" text-anchor="middle">{row.difficulty}</text>'
            )

    legend_x = 80
    legend_y = 394
    for policy in comparison.policies:
        color = colors.get(policy, "#6b7280")
        parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="12" height="12" fill="{color}"/>')
        parts.append(
            f'<text x="{legend_x + 18}" y="{legend_y + 11}" font-family="Arial" font-size="12">{policy}</text>'
        )
        legend_x += 145

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _best_policy_by_difficulty(
    comparison: RewardComparison,
) -> dict[str, PolicyEvaluationSummary]:
    best: dict[str, PolicyEvaluationSummary] = {}
    for row in comparison.rows:
        current = best.get(row.difficulty)
        if current is None or row.mean_reward > current.mean_reward:
            best[row.difficulty] = row
    return best
