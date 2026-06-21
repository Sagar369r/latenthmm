.PHONY: test test-all lint typecheck clean

test:
	pytest v7/v7_engine/tests/ -v --tb=short --cov=v7 --cov-report=term-missing -m "not slow"

test-all:
	pytest v7/v7_engine/tests/ -v --tb=short --cov=v7 -m "not integration"

lint:
	ruff check v7/ api/ data_pipeline/ visualizer/

typecheck:
	mypy v7/ api/ data_pipeline/ visualizer/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name ".pytest_cache" -exec rm -rf {} +
