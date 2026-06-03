#!/usr/bin/env python3
"""train_phase_selector.py — delegates to train_phase_selector_scaffold.py.

Full training loop requires server GPU. The scaffold validates config and prevents
privileged leakage. Rename the scaffold and implement the training loop on GPU server.
"""

if __name__ == "__main__":
    import os, sys
    scaffold = os.path.join(os.path.dirname(__file__), "train_phase_selector_scaffold.py")
    sys.argv[0] = scaffold
    exec(open(scaffold).read())
