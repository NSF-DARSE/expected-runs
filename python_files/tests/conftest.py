"""Make python_files/ importable for every test in this directory.

Without this, test files depended on whichever sibling happened to run first
inserting the path -- so running one file alone, or under -k, failed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
