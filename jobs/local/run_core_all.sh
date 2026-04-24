#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

"$ROOT_DIR/jobs/local/run_core_beta01.sh"
"$ROOT_DIR/jobs/local/run_core_beta05.sh"
