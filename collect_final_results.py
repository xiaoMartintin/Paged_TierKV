#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Optional


COMMANDS = [
    ("Main final benchmark package", "modal run main.py", "main.log"),
    ("Tier-mode ablation", "modal run main.py --tier-mode-ablation", "tier_mode.log"),
    ("Quality sanity", "modal run main.py --quality-sanity", "quality_sanity.log"),
    (
        "Stage A scaling",
        "TIERKV_STAGED_COUNTS=32,64,96,128,160 "
        "TIERKV_STAGED_CONTEXT_TOKENS=1900 "
        "TIERKV_STAGED_DECODE_STEPS=16 "
        "modal run main.py --staged-scaling",
        "stage_a.log",
    ),
    (
        "Stage B scaling",
        "TIERKV_STAGED_COUNTS=192,224,256,320 "
        "TIERKV_STAGED_CONTEXT_TOKENS=1900 "
        "TIERKV_STAGED_DECODE_STEPS=16 "
        "modal run main.py --staged-scaling",
        "stage_b.log",
    ),
    (
        "Boundary run",
        "TIERKV_STAGED_COUNTS=384,448,512 "
        "TIERKV_STAGED_CONTEXT_TOKENS=1900 "
        "TIERKV_STAGED_DECODE_STEPS=16 "
        "modal run main.py --staged-scaling",
        "boundary.log",
    ),
    (
        "7B extension scaling at 1024 tokens",
        "TIERKV_EXTENSION_MODEL_ID=${TIERKV_EXTENSION_MODEL_ID:-lmsys/vicuna-7b-v1.5} "
        "TIERKV_EXTENSION_CONTEXT_TOKENS=1024 "
        "TIERKV_EXTENSION_COUNTS=1,2,4,6,8,12 "
        "TIERKV_EXTENSION_DECODE_STEPS=16 "
        "modal run main.py --extension-scaling",
        "extension_1024.log",
    ),
    (
        "7B extension scaling at 1536 tokens",
        "TIERKV_EXTENSION_MODEL_ID=${TIERKV_EXTENSION_MODEL_ID:-lmsys/vicuna-7b-v1.5} "
        "TIERKV_EXTENSION_CONTEXT_TOKENS=1536 "
        "TIERKV_EXTENSION_COUNTS=1,2,4,6,8 "
        "TIERKV_EXTENSION_DECODE_STEPS=16 "
        "modal run main.py --extension-scaling",
        "extension_1536.log",
    ),
]

SECTIONS = [
    ("1. Clean Sanity Table", "main.log"),
    ("2. Tier-Mode Ablation", "tier_mode.log"),
    ("3. Quality Sanity", "quality_sanity.log"),
    ("4. Stage A Scaling (32,64,96,128,160)", "stage_a.log"),
    ("5. Stage B Scaling (192,224,256,320)", "stage_b.log"),
    ("6. Boundary Run (384,448,512)", "boundary.log"),
]

ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    text = ANSI_RE.sub("", text)
    text = CONTROL_RE.sub("", text)
    return text.replace("\r", "")


def read_log(log_dir: Path, filename: str) -> Optional[str]:
    path = log_dir / filename
    if not path.exists():
        return None
    return clean_text(path.read_text(encoding="utf-8", errors="replace"))


def is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def is_heading_line(line: str) -> bool:
    stripped = line.strip()
    if stripped.startswith("### Command:"):
        return False
    return (
        stripped.startswith("===")
        or stripped.startswith("###")
        or stripped in {
            "No-score path cleanup (post-fix validation):",
            "Default configuration (policy_interval=64):",
            "Memory-realism metrics:",
            "What still limits TierKV performance:",
            "Final best story:",
        }
    )


def normalized_heading(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("==="):
        return stripped.strip("= ").strip()
    if stripped.startswith("###"):
        return stripped.strip("# ").strip()
    return stripped


def extract_relevant_blocks(text: str) -> list[str]:
    lines = text.splitlines()
    blocks: list[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if is_heading_line(line):
            heading = normalized_heading(line)
            if heading:
                blocks.append(f"### {heading}" if not heading.startswith("#") else heading)
            idx += 1
            continue

        if is_table_line(line):
            table = []
            while idx < len(lines) and is_table_line(lines[idx]):
                table.append(lines[idx].rstrip())
                idx += 1
            blocks.append("\n".join(table))
            continue

        if line.startswith("Common successful range:"):
            analysis = []
            while idx < len(lines):
                current = lines[idx].rstrip()
                if not current:
                    break
                if current.startswith("Stopping app") or current.startswith("✓ App completed"):
                    break
                analysis.append(current)
                idx += 1
            blocks.append("\n".join(analysis))
            continue

        idx += 1
    return blocks


def find_heading_index(lines: list[str], heading: str, start_idx: int = 0) -> Optional[int]:
    for idx in range(start_idx, len(lines)):
        if normalized_heading(lines[idx]) == heading:
            return idx
    return None


def extract_heading_block(text: str, heading: str, table_only: bool = False) -> Optional[str]:
    lines = text.splitlines()
    start = find_heading_index(lines, heading)
    if start is None:
        return None

    block = [f"### {heading}"]
    idx = start + 1
    table_started = False
    table_finished = False
    while idx < len(lines):
        line = lines[idx].rstrip()
        if idx > start + 1 and is_heading_line(line):
            break
        if line.startswith("Stopping app") or line.startswith("✓ App completed"):
            break
        if table_only:
            if is_table_line(line):
                table_started = True
            elif table_started:
                table_finished = True
            if table_finished:
                break
        block.append(line)
        idx += 1

    while block and block[-1] == "":
        block.pop()
    return "\n".join(block)


def extract_heading_blocks(text: str, headings: list[str], table_only: bool = False) -> list[str]:
    blocks = []
    for heading in headings:
        block = extract_heading_block(text, heading, table_only=table_only)
        if block:
            blocks.append(block)
    return blocks


def parse_table_rows(text: str, header_marker: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    rows: list[dict[str, str]] = []
    for idx, line in enumerate(lines):
        if header_marker not in line:
            continue
        header = [cell.strip() for cell in line.strip().strip("|").split("|")]
        row_idx = idx + 2
        while row_idx < len(lines) and is_table_line(lines[row_idx]):
            values = [cell.strip() for cell in lines[row_idx].strip().strip("|").split("|")]
            if len(values) == len(header):
                rows.append(dict(zip(header, values)))
            row_idx += 1
        break
    return rows


def first_crossover(rows: list[dict[str, str]]) -> Optional[tuple[str, str]]:
    for row in rows:
        try:
            baseline = float(row.get("Baseline tok/s", ""))
            tierkv = float(row.get("TierKV tok/s", ""))
        except ValueError:
            continue
        if tierkv >= baseline:
            return row.get("Requests", ""), f"{tierkv / max(baseline, 1e-6):.2f}x"
    return None


def boundary_summary(boundary_text: Optional[str]) -> str:
    if not boundary_text:
        return "Boundary log missing; final failure-boundary summary is not available yet."

    rows = parse_table_rows(boundary_text, "| Method | Max Successful Requests |")
    if not rows:
        return "Boundary table not found in boundary log."

    lines = [
        "| Method | Max Successful Requests | Largest Successful Total Context Tokens | Peak MB At Max Success | Failure Request Count | Failure Type |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.get('Method', '')} | "
            f"{row.get('Max Successful Requests', '')} | "
            f"{row.get('Largest Successful Total Context Tokens', '')} | "
            f"{row.get('Peak MB At Max Success', '')} | "
            f"{row.get('Failure Request Count', '')} | "
            f"{row.get('Failure Type', '')} |"
        )
    return "\n".join(lines)


def summarize_tinyllama(boundary_text: Optional[str]) -> list[str]:
    if not boundary_text:
        return [
            "- TinyLlama logs have not been collected yet.",
            "- Run `./run_final_evaluation.sh` to generate raw logs and refresh this report.",
        ]

    throughput_rows = parse_table_rows(boundary_text, "| Requests | Baseline tok/s | TierKV tok/s |")
    memory_rows = parse_table_rows(boundary_text, "| Requests | Baseline Peak MB | TierKV Peak MB |")
    status_rows = parse_table_rows(boundary_text, "| Requests | Baseline status | TierKV status |")

    baseline_oom = any(row.get("Baseline status") == "OOM" for row in status_rows)
    tierkv_ok_at_oom = any(
        row.get("Baseline status") == "OOM" and row.get("TierKV status") == "OK"
        for row in status_rows
    )

    lines = ["- Main TinyLlama result: TierKV uses compact HOT+WARM storage and reports logical and physical KV compression."]

    if memory_rows:
        last_memory = memory_rows[-1]
        lines.append(
            "- Boundary memory point: "
            f"{last_memory.get('Requests')} requests, "
            f"Baseline Peak `{last_memory.get('Baseline Peak MB')}` MB, "
            f"TierKV Peak `{last_memory.get('TierKV Peak MB')}` MB, "
            f"TierKV Physical KV `{last_memory.get('TierKV Physical KV MB')}` MB, "
            f"Physical Compression `{last_memory.get('TierKV Physical Compression')}`."
        )

    if throughput_rows:
        ratios = [row.get("TierKV / Baseline", "") for row in throughput_rows if row.get("TierKV / Baseline")]
        if ratios:
            lines.append(f"- Boundary throughput ratios reported: {', '.join(ratios)}.")

    if baseline_oom and tierkv_ok_at_oom:
        lines.append(
            "- Strongest systems claim: dense baseline hits the boundary first, while TierKV succeeds at the same request count."
        )
    elif status_rows:
        last_status = status_rows[-1]
        lines.append(
            "- Max reported boundary status: "
            f"{last_status.get('Requests')} requests, "
            f"Baseline `{last_status.get('Baseline status')}`, "
            f"TierKV `{last_status.get('TierKV status')}`."
        )
    else:
        lines.append("- Boundary status table was not found; no failure-boundary claim is made.")

    return lines


def summarize_extension(log_texts: list[tuple[str, Optional[str]]]) -> list[str]:
    present_logs = [(label, text) for label, text in log_texts if text]
    if not present_logs:
        return ["- 7B extension logs have not been collected yet."]

    lines = ["- 7B extension result: this is a supporting experiment, not the primary TinyLlama result path."]
    any_crossover = False
    any_rows = False

    for label, text in present_logs:
        rows = parse_table_rows(text or "", "| Requests | Baseline tok/s | TierKV tok/s |")
        memory_rows = parse_table_rows(text or "", "| Requests | Baseline Peak MB | TierKV Peak MB |")
        if not rows:
            lines.append(f"- {label}: throughput table not found.")
            continue
        any_rows = True
        crossover = first_crossover(rows)
        if crossover is not None:
            any_crossover = True
            request_count, ratio = crossover
            lines.append(f"- {label}: TierKV reaches/exceeds baseline at `{request_count}` requests (`{ratio}`).")
        else:
            first_ratio = rows[0].get("TierKV / Baseline", "n/a")
            last_ratio = rows[-1].get("TierKV / Baseline", "n/a")
            lines.append(f"- {label}: no throughput crossover; ratio moves from `{first_ratio}` to `{last_ratio}`.")
        if memory_rows:
            first_mem = memory_rows[0]
            last_mem = memory_rows[-1]
            lines.append(
                f"- {label}: memory gap grows from "
                f"`{_memory_gap(first_mem)}` MB to `{_memory_gap(last_mem)}` MB; "
                f"last physical compression `{last_mem.get('TierKV Physical Compression', 'n/a')}`."
            )

    if any_rows and not any_crossover:
        lines.append(
            "- Extension conclusion: TierKV strengthens the memory-scaling story under a larger MHA model, "
            "but it does not exceed baseline raw decode throughput in the tested extension ranges."
        )
    return lines


def _memory_gap(row: dict[str, str]) -> str:
    try:
        baseline = float(row.get("Baseline Peak MB", ""))
        tierkv = float(row.get("TierKV Peak MB", ""))
    except ValueError:
        return "n/a"
    return f"{baseline - tierkv:.2f}"


def high_level_summary(
    boundary_text: Optional[str],
    extension_1024_text: Optional[str],
    extension_1536_text: Optional[str],
) -> str:
    lines = summarize_tinyllama(boundary_text)
    lines.extend(
        summarize_extension(
            [
                ("1024-token extension", extension_1024_text),
                ("1536-token extension", extension_1536_text),
            ]
        )
    )
    return "\n".join(lines)


def section_from_log(log_dir: Path, filename: str) -> str:
    text = read_log(log_dir, filename)
    if text is None:
        return f"_Log missing or not run yet: `{log_dir / filename}`_"
    blocks = extract_relevant_blocks(text)
    if not blocks:
        return f"_No Markdown tables or analysis blocks found in `{log_dir / filename}`._"
    return "\n\n".join(blocks)


def extension_section_from_log(log_dir: Path, filename: str, headings: list[str], table_only: bool = False) -> str:
    text = read_log(log_dir, filename)
    if text is None:
        return f"_Log missing or not run yet: `{log_dir / filename}`_"
    blocks = extract_heading_blocks(text, headings, table_only=table_only)
    if not blocks:
        return f"_Requested extension blocks were not found in `{log_dir / filename}`._"
    return "\n\n".join(blocks)


def build_report(log_dir: Path) -> str:
    boundary_text = read_log(log_dir, "boundary.log")
    extension_1024_text = read_log(log_dir, "extension_1024.log")
    extension_1536_text = read_log(log_dir, "extension_1536.log")
    lines = [
        "# Final Evaluation Results",
        "",
        "## Environment / Notes",
        "- Codebase is treated as frozen for final evaluation.",
        "- Default TierKV config: `policy_interval=64`, `policy_on_new_block=False`, `policy_mode=hot_warm`.",
        "- Pool layout: compact state-separated HOT/WARM/COLD storage.",
        "- Main result path: TinyLlama staged high-KV-pressure evaluation.",
        "- 7B extension path: Vicuna/LLaMA-compatible MHA experiment, reported separately as supporting evidence.",
        f"- Log directory: `{log_dir}`.",
        "",
        "## Commands Run",
    ]

    for label, command, filename in COMMANDS:
        lines.extend(
            [
                f"### {label}",
                f"Log: `{log_dir / filename}`",
                "```bash",
                command,
                "```",
                "",
            ]
        )

    lines.extend(["## 1. Main TinyLlama Results", ""])

    for title, filename in SECTIONS:
        lines.extend(
            [
                f"### {title}",
                f"Source log: `{log_dir / filename}`",
                "",
                section_from_log(log_dir, filename),
                "",
            ]
        )
        if filename == "quality_sanity.log":
            lines.append(
                "_If this section duplicates the clean sanity output, the current CLI did not emit a distinct standalone quality-only table._"
            )
            lines.append("")

    lines.extend(
        [
            "### 7. Final Failure-Boundary Summary",
            boundary_summary(boundary_text),
            "",
            "## 2. 7B Extension Experiment",
            "The 7B extension is an additional evaluation path and does not replace the TinyLlama main result.",
            "",
            "### 2.1 Extension Model",
            "Source logs: `extension_1024.log`, `extension_1536.log`",
            "",
            extension_section_from_log(log_dir, "extension_1024.log", ["7B Extension Model Config"], table_only=True),
            "",
            "### 2.2 Passkey Sanity",
            "Source logs: `extension_1024.log`, `extension_1536.log`",
            "",
            extension_section_from_log(log_dir, "extension_1024.log", ["7B Extension Passkey Sanity"], table_only=True),
            "",
            "### 2.3 1024-Token Extension Scaling",
            f"Source log: `{log_dir / 'extension_1024.log'}`",
            "",
            extension_section_from_log(
                log_dir,
                "extension_1024.log",
                [
                    "7B Extension Throughput Scaling",
                    "7B Extension Memory Scaling",
                    "7B Extension Status",
                    "7B Extension Failure Boundary",
                    "Scaling Analysis",
                    "7B Extension Crossover Analysis",
                ],
            ),
            "",
            "### 2.4 1536-Token Extension Scaling",
            f"Source log: `{log_dir / 'extension_1536.log'}`",
            "",
            extension_section_from_log(
                log_dir,
                "extension_1536.log",
                [
                    "7B Extension Throughput Scaling",
                    "7B Extension Memory Scaling",
                    "7B Extension Status",
                    "7B Extension Failure Boundary",
                    "Scaling Analysis",
                    "7B Extension Crossover Analysis",
                ],
            ),
            "",
            "### 2.5 Extension Crossover Analysis",
            "",
            extension_section_from_log(log_dir, "extension_1024.log", ["7B Extension Crossover Analysis"]),
            "",
            extension_section_from_log(log_dir, "extension_1536.log", ["7B Extension Crossover Analysis"]),
            "",
            "### 2.6 Extension Conclusion",
            "",
            extension_section_from_log(log_dir, "extension_1024.log", ["Final Extension Conclusion Guidance"]),
            "",
            extension_section_from_log(log_dir, "extension_1536.log", ["Final Extension Conclusion Guidance"]),
            "",
            "## 3. Final High-Level Summary",
            high_level_summary(boundary_text, extension_1024_text, extension_1536_text),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect final TierKV evaluation logs into Markdown.")
    parser.add_argument("--log-dir", default="final_eval_logs/latest")
    parser.add_argument("--output", default="FINAL_EVALUATION_RESULTS.md")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output = Path(args.output)
    output.write_text(build_report(log_dir), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
