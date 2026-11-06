# Retrieve Saved Colab Results

Use this procedure when a Colab connection is interrupted, the download cell reports `NameError: name 'result' is not defined`, or the original experiment folder is absent from the current runtime.

The error establishes that the current Python session lacks the `result` variable. It does not establish whether a result ZIP, raw measurements or a Drive checkpoint still exists. A disconnected browser, a restarted Python kernel and a replaced virtual machine have different consequences; the error alone does not identify which occurred.

## 1. Find the saved copy

1. Open [`Download_Saved_Results.ipynb`](../notebooks/Download_Saved_Results.ipynb) in Colab. A CPU runtime is sufficient.
2. Run its Drive mount and archive-listing cells. Authorize the **same Google account** used for experiment backups.
3. Locate the exact experiment ID recorded by the original notebook. Set `RECOVERY_EXPERIMENT_ID` to that listed ID when multiple experiments exist. It may remain blank when only one saved experiment is available.
4. Leave `RECOVERY_ARCHIVE_NAME` blank to prefer the selected experiment's final results ZIP, falling back to its latest verified checkpoint. To retrieve a specific earlier archive, enter its exact listed filename.
5. Run the download cell. Read the printed archive kind and recorded status, and keep the ZIP unchanged for review. A checksum failure is reported rather than silently selecting another file.

The recovery notebook retrieves files; it does not run GPU measurements, train another model or automatically resume an interrupted experiment. It does not depend on Python variables from the original notebook.

The default Drive location is:

```text
My Drive/Transformer_Linear_Layer_Performance_Predictor/experiments
```

You can also navigate to that folder directly in Google Drive and download the selected ZIP. If you changed the backup directory, use the directory printed by the original experiment instead.

## 2. Understand which evidence survived

| What is available | What it establishes | Next step |
|---|---|---|
| A final results ZIP | Export reached the point at which an archive was written and saved; the experiment itself may have completed or failed | Read `launcher_status.json`, `run/status.json` and `run/report.html` before interpreting scores |
| A checkpoint after primary analysis | The snapshot may contain completed primary results while later stages remain unfinished | Inspect the primary dataset audit and analysis status; do not describe the entire experiment as complete |
| A checkpoint after timing but before profiling | Timing records and environment information may be retained, while diagnostics and final validation are incomplete | Preserve the archive for a separate recovery assessment; normal completed-dataset analysis can reject it |
| A checkpoint after an earlier timing visit | Only the work recorded up to that boundary is retained | Keep the raw visits and recorded status; do not invent the missing visits or targets |
| Notebook text or screenshots only | Displayed messages survive, but the full measurement records are not established | Check downloaded files and the backup folder before deciding whether a new experiment is necessary |

The checkpoint metadata identifies the snapshot boundary and retained files. A valid checksum establishes file consistency with that snapshot. It does not certify hardware timing, restore missing observations or turn a partial run into a completed experiment.

## 3. If no saved archive appears

First confirm the Google account, backup directory and experiment ID. Check the computer's Downloads folder for an archive already downloaded during the original session. If the original runtime is still accessible, its local experiment folder may provide another recovery source.

An empty `/content` search only describes the currently connected runtime. It cannot establish whether another copy exists in Drive or on the local computer. However, **if the original runtime has been deleted and no copy was saved elsewhere, its raw measurements cannot be recovered from notebook output**. A saved notebook contains its code and displayed output, not an automatic copy of every file created in the runtime. See the [Colab FAQ](https://research.google.com/colaboratory/faq.html).

In that case, a fresh measurement attempt needs a new experiment ID. Keep any surviving error messages as context, and establish the Drive checkpoint destination before starting the new attempt.

## 4. Why the timing progress can finish before the experiment

The primary progress message `visit 3/3: 609/609 shapes` marks completion of the scheduled timing pass. In this implementation, successful raw visits and completed aggregates have already been written locally by that point. Separate profiler captures, final dataset validation, model analysis, the repeat dataset, comparison and export can still remain.

The notebook therefore saves a checkpoint **before profiling begins** as well as at full timing-visit boundaries and later experiment stages. Profiling progress is reported separately. A profiler warning alone is not enough to diagnose a crash; inspect the saved status and `worker.log`.

## 5. Preserve the experiment's meaning

- Keep original archives and their recorded status. Do not mark an incomplete dataset `completed` to bypass validation.
- Keep measurement attempts separate. A new runtime can have a different GPU, software environment or operating state.
- Do not concatenate incomplete visits with observations from another GPU and label them one uninterrupted experiment.
- Treat a later recovery analysis as a derived artifact linked to the retained evidence. Record missing diagnostics and any changed assumptions.
- Treat a completed Drive copy as a recovery point, not a guarantee against all storage failures. Loss before a checkpoint finishes can still leave unbacked work.

For the fresh-experiment workflow, see [START_HERE.md](START_HERE.md). For the measurement definitions, see [PROTOCOL.md](PROTOCOL.md).
