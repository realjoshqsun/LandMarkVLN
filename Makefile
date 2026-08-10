PYTHON ?= python3
export PYTHONPATH := $(CURDIR)

.PHONY: verify reproduce-analysis visualize audit-anonymity test artifact

verify:
	$(PYTHON) scripts/verify_frozen_outputs.py

reproduce-analysis: verify
	$(PYTHON) scripts/reproduce_analysis.py
	$(PYTHON) scripts/render_readme_visuals.py

visualize:
	$(PYTHON) scripts/render_readme_visuals.py

audit-anonymity:
	$(PYTHON) scripts/audit_anonymity.py

test:
	$(PYTHON) -m unittest discover -s tests -v

artifact: reproduce-analysis audit-anonymity test
	$(PYTHON) scripts/build_anonymous_artifact.py

