# Step A Repeat Artifact Reuse

## Measurement basis
- Profiling assessment identified Step A contact-sheet generation as the top optimization target.
- Observed baseline: `window_repeat_count=1` total runtime about `14.1s`; `window_repeat_count=2` about `28.1s` (`1.989x`).
- The duplicated `ffmpeg` contact-sheet extraction accounted for `97%+` of the extra repeated-window cost.

## Change
- Added server-side reuse for Step A repeated window jobs in `src/video2tasks/server/app.py`.
- The first repeat that materializes a `SharedFSImageTransport` for a contact-sheet batch seeds a small in-memory cache.
- Later repeats for the same logical window reuse that exact transport instead of calling `extractor.get_many_b64_with_artifacts(...)` again.
- Cache scope is intentionally narrow: same subset/sample/job type/window pass/window id, same `frame_ids`, same artifact kind, and same contact-sheet shape knobs.
- Inline-image paths are unchanged. Task ids, dispatch ids, retries, and result envelopes remain per-repeat and independent.

## Expected benefit
- Removes the repeated Step A contact-sheet extraction work for repeat jobs while preserving repeat protocol semantics.
- For `window_repeat_count=2`, the second repeat should no longer pay the dominant `ffmpeg` extraction cost for the same logical window.
- Expected outcome: Step A extraction stops scaling nearly linearly with repeat count; total wall time should move closer to `repeat=1` extraction cost plus the extra model-call overhead.
