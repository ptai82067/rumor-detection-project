#!/usr/bin/env python3
"""
Rumor Detection using PHEME + Knowledge Graph

Main entry point for the rumor detection project.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def main():
    """Main entry point."""
    print("Rumor Detection Project")
    print("=======================")
    print("This project implements rumor detection using PHEME dataset")
    print("and FakeNewsNet with knowledge graph integration.")
    print()
    print("Available modules:")
    print("- preprocessing: Data preprocessing pipeline")
    print("- models: Machine learning models")
    print("- knowledge_graph: Knowledge graph construction")
    print("- notebooks: Jupyter notebooks for analysis")
    print()
    print("Run 'python -m preprocessing.run_pipeline' to start preprocessing")
    print("Run 'jupyter notebook' to explore analysis notebooks")

if __name__ == "__main__":
    main()