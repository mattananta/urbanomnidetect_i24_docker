#!/usr/bin/env bash
set -euo pipefail
cd /opt/urbanomnidetect
exec python /opt/urbanomnidetect/evaluate_urbanomnidetect_i24.py "$@"
