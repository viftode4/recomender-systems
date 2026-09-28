# Final verification

The single reserved batch completed all eight frozen evaluation jobs and five
fixed analyses. No model, checkpoint, parameter or comparison was chosen after
test access. `aggregate.json` records each output manifest hash.

The final source suite discovered 216 tests: 211 passed in the isolated training
environment and five PDF tests were skipped there because it has no rendering
stack. All 23 report tests, including those five, passed separately in the local
rendering environment. There were no failures.

The seven-page PDF has Times New Roman 12pt and 1.15 spacing, with a cover, three
main task pages and three research appendices. Main discussion lengths are
94, 122 and 113 words. Every actual page was visually inspected. The PDF hash is
recorded in `aggregate.json`.

The six-scenario offline demo uses a hand-authored synthetic profile. Functional
controls and exported model predictions passed verification. Chrome aborted
before page loading, so browser visual verification is not claimed.

Archive hashes, member checks and extracted-package tests are verified after
assembly; the receipt is stored beside the archive rather than inside it.
Verification uses the local macOS environments and does not establish successful
installation on every operating system. The report remains a review draft until
the group supplies real names and contributions; peer feedback remains individual
human work. No course submission or GitHub push has occurred.
