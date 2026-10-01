# Shareable packages

| Package | Purpose |
| --- | --- |
| [coursework-complete-v2/24.zip](coursework-complete-v2/24.zip) | Verified coursework code/report snapshot |
| `team-meeting-2026-10-01/Group24-team-start.zip` | Generated offline HTML guide with the coursework snapshot |

The coursework snapshot's [verification receipt](coursework-complete-v2/verification.json)
records its checks. It is a review draft, not a submission receipt.

To rebuild the current team guide, follow the [quickstart](../docs/QUICKSTART.md),
then run from the repository root:

```sh
final-project/.venv/bin/python -m pip install --index-url https://pypi.org/simple -r final-project/requirements-handoff.txt
make handoff
```

Open `final-project/packages/team-meeting-2026-10-01/preview/START_HERE.html`
or share the generated ZIP. The dated directory is the meeting-pack destination;
it reads current guides from `docs/` and `docs/team/`.

The generated ZIP and preview are ignored by Git and can be rebuilt. The original
coursework ZIP is tracked and preserved by the builder. Packaging starts no training.
