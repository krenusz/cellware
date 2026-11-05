"""E48 readings: the E46 analysis applied to E48; then `b29_rescore.py e46 e48` for the fit-set
primary and the B7 CRPS comparison (amended protocol).

    uv run python experiments/analyze_e48.py
"""

import analyze_e46
import e48_capsel6 as e48

if __name__ == "__main__":
    analyze_e46.main(name="e48", sim_mod=e48)
