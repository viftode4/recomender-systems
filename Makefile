PYTHON ?= final-project/.venv/bin/python

.PHONY: help check handoff

help:
	@echo "make check    Run the fast synthetic checks; no data or training"
	@echo "make handoff  Build the meeting ZIP; requires requirements-handoff.txt"
	@echo "Override the interpreter with: make check PYTHON=/path/to/python"

check:
	"$(PYTHON)" final-project/operations/team_smoke_check.py

handoff:
	"$(PYTHON)" final-project/operations/package_team_meeting.py
