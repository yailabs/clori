#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
python3 tests/c_structure.py self-test
python3 tests/c_structure.py check natural
