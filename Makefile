.PHONY: lint weights

METHODS := least-squares bradley-terry-regularized huber-log elo pagerank

lint:
	uvx ruff check
	uvx ty check

weights:
	mkdir -p data/weights
	for method in $(METHODS); do \
		uv run --env-file .env compute_weights.py --csv data/runs --method $$method --output data/weights/$$method.csv; \
	done
