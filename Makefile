env:
	python3 -m venv venv

install:
	. ./venv/bin/activate && python3 -m pip install -r requirements.txt

type-checks:
	mypy .

run:
	. ./venv/bin/activate && python3 app.py

clean:
	rm -rf __pycache__
	rm -rf **/__pycache__/
	deactivate
	rm -rf ./venv
	rm -rf .mypy_cache

init: env install

.PHONY: env install init clean

