"""E47 readings, as pre-registered: the E46 analysis applied to E47 (gate, gamma, B27, B24/B26,
instruction shifts), followed by the B29 fit-set re-score with CRPS (b29_rescore.py e47).

    uv run python experiments/analyze_e47.py
"""

import analyze_e46
import e47_capsel as e47

if __name__ == "__main__":
    analyze_e46.main(name="e47", sim_mod=e47)
