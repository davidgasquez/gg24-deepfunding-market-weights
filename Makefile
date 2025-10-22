.PHONY: lint weights

METHODS := least-squares bradley-terry-regularized huber-log elo pagerank

lint:
	uvx ruff check
	uvx ty check

weights:
	mkdir -p data/phase_1/weights
	for method in $(METHODS); do \
		uv run --env-file .env compute_weights.py --csv data/phase_1/runs --method $$method --output data/phase_1/weights/$$method.csv; \
	done
	mkdir -p data/phase_2/weights
	for method in $(METHODS); do \
		uv run --env-file .env compute_weights.py --csv data/phase_2/runs --method $$method --output data/phase_2/weights/$$method.csv; \
	done
