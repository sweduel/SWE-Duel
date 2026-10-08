.PHONY: setup-repos build-docker test test-gates install

install:
	./venv/bin/pip install -e ".[dev]"

setup-repos:
	./venv/bin/swe-duel-setup repos

build-docker:
	./venv/bin/swe-duel-setup docker

test:
	./venv/bin/pytest tests/ -v

# Long Docker-backed gate regression suite (auto-parallel via pytest-xdist).
# Override workers: `make test-gates WORKERS=8` or `SWE_DUEL_GATE_TEST_WORKERS=8`.
WORKERS ?=
test-gates:
	SWE_DUEL_GATE_TEST_WORKERS="$(WORKERS)" ./venv/bin/pytest tests/test_validation_gates_regression.py -v
