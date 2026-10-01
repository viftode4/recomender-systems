PYTHON ?= final-project/.venv/bin/python

.PHONY: help check check-docs handoff

help:
	@echo "make check       Check documentation and run fast synthetic tests"
	@echo "make check-docs  Check active Markdown links and headings; stdlib only"
	@echo "make handoff     Build the meeting ZIP; requires requirements-handoff.txt"
	@echo "Override the interpreter with: make check PYTHON=/path/to/python"

check: check-docs
	"$(PYTHON)" final-project/operations/team_smoke_check.py

check-docs:
	"$(PYTHON)" final-project/operations/check_docs.py

handoff:
	"$(PYTHON)" final-project/operations/package_team_meeting.py
