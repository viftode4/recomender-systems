# Synthetic mechanism check

The first run used 600 FIT candidates, 600 inner PROBE candidates, and 5,000 evaluation candidates per task. Outcomes had 6% independent label noise. Candidate relations were supplied synthetic features, not learned movie representations. The FIT search produced at most 112 calibrated trees; each final choice was serialized before its evaluation sample was created.

| Task | Selected single expression | Program evaluation NLL | Fixed initial expression NLL | Signed linear + pair features NLL |
|---|---|---:|---:|---:|
| Conjunction amid noise | `distinct(a0*a1)` | 0.2331 | 0.5922 | 0.2364 |
| Redundant equivalent branches | `a0` | 0.2492 | 0.2492 | 0.2528 |
| XOR outside the monotone grammar | `max(a0,a1)` | 0.5494 | 0.6938 | **0.2655** |

The conjunction's function was recovered on unseen candidates. Removing a redundant branch reduced the program from three nodes to one with identical predictions; it did not improve accuracy. The XOR failure shows a meaningful expressiveness limit: the reference's signed combination represents a relation that this positive max/product tree cannot express.

These are one-seed synthetic diagnostics, not evidence of recommender performance or a novel research contribution. The modest difference from the signed linear/pair reference on the successful tasks is not a superiority claim. The proof establishes that the implementation performs structural edits and can select a supplied composition, while preserving a genuine out-of-sample check.

Raw reproducibility artifacts are in the ignored `runs/structural-program-synthetic-v1/` directory. Eight unit tests passed, covering distinct-witness correctness against exhaustive enumeration, cloned-evidence rejection, self-evidence rejection, empty contexts, structural edits, FIT/PROBE separation, functional recovery, redundant branch deletion, serialization, and permutation behavior.
