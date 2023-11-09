PYTHON := poetry run python
PYSEN := poetry run pysen
DVSB_CONFIG_NAME := ${DVSB_CONFIG_NAME}


.PHONY: lint
lint:
	@${PYSEN} run lint


.PHONY: format
format:
	@${PYSEN} run format


.PHONY: install
install:
	poetry install


.PHONY: run_benchmark
run_benchmark:
ifndef DVSB_CONFIG_NAME
	@PYTHONPATH=. ${PYTHON} tools/run_benchmark.py
else
	@PYTHONPATH=. ${PYTHON} tools/run_benchmark.py -n ${DVSB_CONFIG_NAME}
endif
