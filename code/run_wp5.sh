#!/bin/bash
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "### stage main (non-Mamba, 310 checkpoints)"; ${PYTHON:-python} wp5_consequence.py || exit 1
echo "### stage mamba (40 checkpoints)"; ${PYTHON_MAMBA:-python} wp5_consequence.py --mamba || exit 1
echo "### WP5 CHAIN DONE"
