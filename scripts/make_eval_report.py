#!/usr/bin/env python3
"""Create a small HTML index for GAP/RoboTwin evaluation videos."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path


def natural_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def find_eval_dirs(root: Path) -> list[Path]:
    if (root / "_result.txt").exists():
        return [root]
    return sorted({p.parent for p in root.rglob("_result.txt")})


def read_result(eval_dir: Path) -> str:
    result_path = eval_dir / "_result.txt"
    if not result_path.exists():
        return "No _result.txt found."
    return result_path.read_text(encoding="utf-8", errors="replace").strip()


def render_eval_dir(eval_dir: Path, report_path: Path) -> str:
    videos = sorted(eval_dir.glob("episode*.mp4"), key=natural_key)
    result = html.escape(read_result(eval_dir))
    rel_dir = html.escape(str(eval_dir.relative_to(report_path.parent)))
    cards = []
    for video in videos:
        rel_video = html.escape(str(video.relative_to(report_path.parent)))
        cards.append(
            f"""
            <article class="video-card">
              <h3>{html.escape(video.stem)}</h3>
              <video controls preload="metadata" src="{rel_video}"></video>
            </article>
            """
        )
    video_html = "\n".join(cards) if cards else "<p>No mp4 videos found in this evaluation directory.</p>"
    return f"""
    <section class="eval-block">
      <h2>{rel_dir}</h2>
      <pre>{result}</pre>
      <div class="video-grid">
        {video_html}
      </div>
    </section>
    """


def build_report(root: Path, output: Path) -> None:
    eval_dirs = find_eval_dirs(root)
    if not eval_dirs:
        raise SystemExit(f"No _result.txt found under {root}")

    output.parent.mkdir(parents=True, exist_ok=True)
    sections = "\n".join(render_eval_dir(eval_dir, output) for eval_dir in eval_dirs)
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GAP Evaluation Report</title>
  <style>
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f7fb;
      color: #172033;
    }}
    header {{
      padding: 24px 32px 8px;
    }}
    h1 {{
      margin: 0;
      font-size: 28px;
      font-weight: 700;
    }}
    .eval-block {{
      margin: 20px 32px 32px;
      padding: 20px;
      background: #ffffff;
      border: 1px solid #d9e0ec;
      border-radius: 8px;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 17px;
      overflow-wrap: anywhere;
    }}
    pre {{
      margin: 0 0 18px;
      padding: 12px;
      border-radius: 6px;
      background: #101828;
      color: #f8fafc;
      white-space: pre-wrap;
    }}
    .video-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 16px;
    }}
    .video-card {{
      min-width: 0;
    }}
    h3 {{
      margin: 0 0 8px;
      font-size: 14px;
      font-weight: 600;
    }}
    video {{
      width: 100%;
      aspect-ratio: 4 / 3;
      border-radius: 6px;
      background: #000;
    }}
  </style>
</head>
<body>
  <header>
    <h1>GAP Evaluation Report</h1>
  </header>
  {sections}
</body>
</html>
"""
    output.write_text(page, encoding="utf-8")
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="results", help="Result root or a single eval directory.")
    parser.add_argument("--output", default=None, help="HTML path. Defaults to <root>/index.html.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else root / "index.html"
    build_report(root, output)


if __name__ == "__main__":
    main()
