.PHONY: help check validate-fast test test-aura

PYTHON ?= python3

help:
	@echo "check          Compile all Python sources"
	@echo "validate-fast Run CPU-only mathematical validation scripts"
	@echo "test           Run the pytest suite"
	@echo "test-aura      Run Aura tests (requires aura/requirements.txt)"

check:
	$(PYTHON) -m compileall -q model experiments scripts tools aura

validate-fast:
	$(PYTHON) -m experiments.validation.back_sym_test
	$(PYTHON) -m experiments.validation.validate_backward_minimal
	$(PYTHON) -m experiments.validation.validate_broadband_minimal
	$(PYTHON) -m experiments.validation.validate_coupled_iteration
	$(PYTHON) -m experiments.validation.validate_multicolor
	$(PYTHON) -m experiments.validation.validate_uncertainty_diffusion
	$(PYTHON) -m experiments.validation.validate_wind_inversion
	$(PYTHON) -m experiments.validation.validate_winged_alignment

test:
	$(PYTHON) -m pytest

test-aura:
	$(PYTHON) -m pytest aura/tests
