#!/bin/bash
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; L=../logs
${PYTHON:-python} wp5b2_selective.py > $L/wp5b2.log 2>&1
${PYTHON_MAMBA:-python} wp5b2_selective.py --mamba >> $L/wp5b2.log 2>&1
echo '### WP5b2 CHAIN DONE' >> $L/wp5b2.log
