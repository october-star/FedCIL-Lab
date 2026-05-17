#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

"$ROOT_DIR/jobs/local/run_reproduce_cifar10_5task_multiseed.sh"
"$ROOT_DIR/jobs/local/run_reproduce_cifar100_10task_multiseed.sh"
