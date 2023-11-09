.PHONY: lint
lint:
	@poetry run pysen run lint


.PHONY: format
format:
	@poetry run pysen run format
