#!/usr/bin/env python3
"""
Runner for all 4 visualization figures (A → D).
Runs each plot script sequentially and reports progress.
"""

import os
import sys
import time
import subprocess

SCRIPTS = [
    ("A", "Class Hierarchy", "plot_class_hierarchy.py"),
    ("B", "Full Ontology Diagram", "plot_ontology_diagram.py"),
    ("C", "KG Sample Subgraph", "plot_kg_sample.py"),
    ("D", "KG Statistics Chart", "plot_kg_statistics.py"),
]

OUTPUT_DIR = os.path.join("docs", "figures")
SCRIPTS_DIR = os.path.join("utils", "visualization")


def get_file_size(path):
    """Get file size in bytes, or 0 if file doesn't exist."""
    if os.path.exists(path):
        return os.path.getsize(path)
    return 0


def main():
    print("=" * 60)
    print("  Visualization Runner — Sinh 4 hình cho báo cáo")
    print("=" * 60)
    print()

    start_time = time.time()
    output_files = []
    success_count = 0

    for letter, name, script in SCRIPTS:
        script_path = os.path.join(SCRIPTS_DIR, script)
        print(f"[{letter}] {name}... ", end="", flush=True)

        t0 = time.time()
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, cwd=os.getcwd(),
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            success_count += 1
            print(f"OK ({elapsed:.1f}s)")

            # Extract output filename from print
            for line in result.stdout.strip().split("\n"):
                if "saved:" in line:
                    path_part = line.split("saved: ")[-1].strip()
                    output_files.append(path_part)
                    print(f"       {line.strip()}")
                elif "Size:" in line:
                    print(f"       {line.strip()}")
                elif "Thread ID:" in line:
                    print(f"       {line.strip()}")
                elif "Posts:" in line:
                    print(f"       {line.strip()}")
                elif "Edges:" in line:
                    print(f"       {line.strip()}")
                elif "Label distribution" in line:
                    print(f"       {line.strip()}")
        else:
            print(f"FAILED ({elapsed:.1f}s)")
            print(f"       Error: {result.stderr.strip()[:200]}")
            print(f"       stdout: {result.stdout.strip()[:200]}")

        print()

    total_elapsed = time.time() - start_time

    print("=" * 60)
    print(f"  Kết quả: {success_count}/{len(SCRIPTS)} thành công")
    print(f"  Thời gian: {total_elapsed:.1f}s")
    print()

    print("  Files saved to docs/figures/:")
    for fpath in sorted(output_files):
        fsize = get_file_size(fpath)
        print(f"    {os.path.basename(fpath):40s} {fsize:>8,} bytes")

    print()
    if success_count == len(SCRIPTS):
        print("=== DONE ===")
        return 0
    else:
        print("=== COMPLETED WITH ERRORS ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())