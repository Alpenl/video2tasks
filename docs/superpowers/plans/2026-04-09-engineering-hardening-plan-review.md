# Engineering Hardening Plan Review

## Review Result
Status: Approved

The revised plan now matches the stated proposal context and current repo state well enough to execute. It builds on already-landed work instead of re-planning from zero, preserves the single-machine shared-filesystem deployment contract, and turns the previously missing runtime contracts into explicit owned tasks with concrete verification paths.

## Blocking Issues

None.

The prior blocking issues are now resolved:

- The .DONE / .FAILED contract is now explicitly owned in Task P0-3, with config-matrix coverage for required stages, optional stages, failure interaction, and finalize/error closure.
- The Stage 2 artifact-layer contract is now explicit in P1-1: Stage 2 is controlled by Stage 2 config, writes official result artifacts even when export is off, and export is reduced to a consumer of Stage 2 output.
- The runtime-correctness stop-the-bleeding work is now pulled into an owned early wave via P0-3, including bad-artifact rejection, bounded empty-result retry behavior, and terminal-state closure.

## Advisory Recommendations

- Keep a single owner on src/video2tasks/server/app.py and src/video2tasks/server/sample_store.py across both P0-3 and P0-4, even if helper work is prepared in parallel. The plan already says this in substance; execution should follow it strictly.
- When dispatching Bundle C, hand subagents the Gate 1 deprecation decision for segments.json.diagnostics up front. That will avoid test lanes encoding different assumptions during the migration window.
- Treat the README and README_CN edits as one coordinated deliverable across P0-1, P0-3, P0-4, P1-1, and P1-2. The plan already marks them as hotspots; enforcing that in execution will keep the documentation from drifting between waves.

## Parallel Execution Assessment

The plan is now parallel-ready.

The serial-gate model is clear and sufficient:

- Gate 0 freezes the smoke surface before documentation and test lanes spread.
- Gate 1 now freezes both runtime evidence destinations and the segments.json.diagnostics compatibility policy, which removes the prior cross-lane ambiguity.
- Gate 2 protects the Stage 2 facade before deeper llm_merge.py and app.py refactors.

The owner model is also clear enough for worker execution:

- app.py and sample_store.py remain explicit single-owner hotspots.
- README.md and README_CN.md remain explicit single-owner hotspots.
- Bundle B now explicitly reiterates that sample_store.py stays single-owner.
- The added P0-3 task cleanly centralizes the runtime semantics that previously would have been easy to miss or split across multiple agents.

Given those changes, subagents should be able to work in parallel without getting stuck on unresolved contract questions or repeatedly colliding on hidden ownership boundaries.

## Suggested Next Move

Dispatch the plan as written, but execute in this order:

1. Complete Gate 0 naming and smoke-surface freeze.
2. Land P0-3 before or tightly alongside P0-4, with one runtime owner controlling app.py and sample_store.py.
3. Freeze Gate 1 before starting broader behavior-test migration or README result-contract edits.
4. Move to P1-1 only after the runtime terminal-state semantics and operator-evidence artifacts are stable.

That sequence should let parallel subagents move quickly without reopening contract decisions during implementation.
