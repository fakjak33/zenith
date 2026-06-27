"""Free data adapters for CAS. Each returns plain Python structures and records
a status dict; all degrade gracefully (return empty + status) rather than raise,
so one flaky source never breaks the whole compute run."""
