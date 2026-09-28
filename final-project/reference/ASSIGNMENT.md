# Verified assignment sources

Retrieved directly through the Brightspace MCP on 2026-09-28.
Course: 844485 (DSAIT4335).

## Final project (Report + Codes), assignment 173405

The assignment entry instructs the group to submit a PDF report and a directory
of runnable code together in a ZIP named after the group number, with one
submission per group. It directs students to the attached slides for detailed
report and submission instructions.

Attached file: [Project-RecSys.pdf](Project-RecSys.pdf), file ID 11701780,
151364 bytes, 19 pages. Downloaded from the assignment attachment, not inferred
from the course-content entry. The extracted instructions match the course
slides previously read: RecBole, MovieLens 100K, ranking, required hybrid models,
independently implemented metrics, group analysis, and rerankers.

The assignment API specifies 2026-10-26T22:59:59Z, which is October 26 at
23:59:59 Europe/Amsterdam. The attached PDF states October 26, 23:59.
No assessment rubric is attached in the current API response.

## Final Project (Peer Feedback), assignment 173406

The entry instructs each student to download the Excel file and provide feedback
on peers. It states that the workbook contains four sheets, one per teammate.
Attachment metadata: PeerFeedback.xlsx, file ID 11701809, 15467 bytes.
The workbook itself has not been downloaded or inspected in this source check.
The same deadline appears in the assignment API.

## Implications for the research direction

- Required regression-weighted and class-taught hybrids cannot be replaced by a
  custom standalone model.
- Task 2.6 explicitly invites insights leading to novel and superior hybrids.
- Task 2.4 invites insightful analysis; Task 2.5 requires user/item group analysis.
- Task 3 requires diversification, calibration, user-side and item-side fairness,
  trade-offs, ordering comparisons, and group impacts.
- The PDF does not explicitly require binarizing all ratings as positive, nor
  explicitly approve a changed rating-aware evaluation protocol. Treat the
  current binarization as an implementation choice and inspect instructor
  configurations before changing the core evaluation. A supplementary
  rating-aware study must be labelled and compared fairly.
- Page 16 prioritizes depth of analysis over the number of models or metrics.

Sources: [assignment entries](https://brightspace.tudelft.nl/d2l/lms/dropbox/user/folders_list.d2l?ou=844485)
and the downloaded attachment above.

Peer-feedback template was downloaded directly from assignment 173406, attachment
11701809, on September 28, 2026: `PeerFeedback.xlsx`, 15,467 bytes. It contains four
member sheets and requires actual retrospective contribution assessments. No ratings
were filled. Course attachments remain local and are excluded from Git and the group
code package.
