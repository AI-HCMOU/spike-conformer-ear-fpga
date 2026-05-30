# SpikeConformer Makefile
# Automation for training, evaluation, conversion, and FPGA synthesis

PYTHON := python
CONFIG := config/default.yaml
CHECKPOINT := checkpoints/best.pth
GPU := 0

.PHONY: all train evaluate convert snn-eval download test lint clean help

help:
	@echo "SpikeConformer - Available targets:"
	@echo "  make download    - Download EarVN1.0 dataset"
	@echo "  make train       - Train Conformer backbone"
	@echo "  make evaluate    - Evaluate ANN model"
	@echo "  make convert     - Convert ANN to SNN"
	@echo "  make snn-eval    - Evaluate SNN model"
	@echo "  make test        - Run unit tests"
	@echo "  make lint        - Run linting"
	@echo "  make fpga        - Run FPGA synthesis (requires Vivado)"
	@echo "  make clean       - Remove generated artifacts"
	@echo ""
	@echo "Configuration:"
	@echo "  CONFIG=$(CONFIG)"
	@echo "  CHECKPOINT=$(CHECKPOINT)"
	@echo "  GPU=$(GPU)"

# Download dataset
download:
	$(PYTHON) scripts/download_data.py --output data/EarVN1.0

# Train the model
train:
	$(PYTHON) scripts/train.py --config $(CONFIG) --gpu $(GPU)

# Evaluate ANN
evaluate:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG) --checkpoint $(CHECKPOINT) --gpu $(GPU)

# Convert ANN -> SNN
convert:
	$(PYTHON) scripts/convert.py --config $(CONFIG) --checkpoint $(CHECKPOINT) --gpu $(GPU)

# Evaluate SNN
snn-eval:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG) --checkpoint $(CHECKPOINT) --snn --gpu $(GPU)

# Run tests
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

# Lint
lint:
	$(PYTHON) -m flake8 src/ scripts/ --max-line-length 100 --ignore E501,W503
	$(PYTHON) -m mypy src/ --ignore-missing-imports

# FPGA synthesis (requires Vivado on PATH)
fpga:
	vivado -mode batch -source hardware/build.tcl

# Clean generated files
clean:
	rm -rf checkpoints/ runs/ hardware/output/
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

# Full pipeline
all: download train evaluate convert snn-eval
