# Refresh Resolution-Stage Lengths Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent Anima multi-resolution DataLoaders from indexing bucket batches that no longer exist after a stage rebuild or cached-stage restore.

**Architecture:** A DatasetGroup inherits PyTorch's `ConcatDataset`, which caches child-dataset cumulative lengths. Resolution stages rebuild or restore bucket indices, changing child lengths. Refresh that cache after either operation so the DataLoader sampler and dataset index mapping agree.

**Tech Stack:** Python, PyTorch `ConcatDataset`, unittest.

---

### Task 1: Regression test and minimal fix

**Files:**
- Modify: `tests/test_anima_resolution_schedule.py`
- Modify: `library/anima_resolution_schedule.py`

**Step 1: Write the failing test**

Create a group with a deliberately stale `cumulative_sizes`, rebuild a stage whose bucket count changes, and assert the cumulative size equals the current child length.

**Step 2: Run test to verify it fails**

Run: `venv\Scripts\python.exe -m unittest tests.test_anima_resolution_schedule.ResolutionScheduleTest.test_stage_rebuild_refreshes_concat_dataset_lengths`

Expected: FAIL because the stale cumulative size is unchanged.

**Step 3: Write minimal implementation**

Add a small helper that invokes `DatasetGroup.cumulative_sizes`' existing cache refresh mechanism after a stage is rebuilt or restored.

**Step 4: Run tests to verify they pass**

Run: `venv\Scripts\python.exe -m unittest tests.test_anima_resolution_schedule`

Expected: PASS.

**Step 5: Commit**

```powershell
git add library/anima_resolution_schedule.py tests/test_anima_resolution_schedule.py docs/plans/2026-08-22-refresh-resolution-stage-lengths.md
git commit -m "fix: refresh dataset lengths for resolution stages"
```
