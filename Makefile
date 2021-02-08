sourcedir = file_processor/ tests/ run.py
testdir = tests/

requirements:
	pip install -r requirements.txt

ci:
	python -m isort --check-only --diff --recursive --quiet $(sourcedir)
	python -m flake8 $(sourcedir)
	python -m pytest $(testdir)
