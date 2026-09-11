.PHONY: install test reproduce clean
install:        ## editable install with the table/figure dependencies
	pip install -e .
test:           ## unit tests
	python3 -m pytest -q
reproduce:      ## regenerate every paper fragment and figure, diff against paper/reference/
	python3 scripts/reproduce_paper.py
clean:
	rm -f paper/generated/*.tex paper/figure/*.pdf paper/figure/*.tex
	find . -name __pycache__ -type d -exec rm -rf {} +
