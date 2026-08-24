PYTHON ?= python3

.PHONY: validate test quick reproduce verify-paper

validate:
	$(PYTHON) scripts/validate_data.py

test:
	$(PYTHON) -m unittest discover -s tests -v

quick:
	$(PYTHON) scripts/run_analysis.py \
		--output-dir artifacts/reproduced/quick \
		--bootstrap-repetitions 10 \
		--randomization-repetitions 20 \
		--spearman-repetitions 10

reproduce:
	$(PYTHON) scripts/run_analysis.py

verify-paper:
	LANDMARKVLN_FULL_TEST=1 $(PYTHON) -m unittest discover -s tests -p 'test_paper_compatibility.py' -v
