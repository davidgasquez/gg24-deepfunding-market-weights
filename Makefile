.PHONY: lint weights

lint:
	uvx ruff check
	uvx ty check

weights:
	mkdir -p data/phase_1/weights
	uv run --env-file .env 03_weights.py --csv data/phase_1/runs --out data/phase_1/weights
	mkdir -p data/phase_2/weights
	uv run --env-file .env 03_weights.py --csv data/phase_2/runs --out data/phase_2/weights --bt-ridge 5
