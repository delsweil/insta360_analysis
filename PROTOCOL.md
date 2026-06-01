# Protocol

## Learned Mistakes

- Do not update a Kalman tracker every frame with a stale detection from the last detection window. A detection should be consumed once; prediction-only steps should advance the filter on intervening frames.
- Do not select `h264_nvenc` only because FFmpeg lists the encoder. Check for a usable NVIDIA runtime first, otherwise Windows machines without NVIDIA GPUs fail at encode time.
- Keep long-running autopan jobs out of request/response handlers. Start them with redirected logs, retain PID/job metadata, and inspect logs/status while they run.
- When merging refactored modules back into the production script, run syntax checks on both the monolithic script and modular package. The v5 refactor had a constructor comma bug that made the module non-executable.
- When gating multiple ball candidates from the same frame, do not let lower-confidence same-timestamp candidates overwrite the physical plausibility gate state established by the best candidate.
- For ball metrics, keep proxy metrics clearly separate from ground-truth metrics. Recall/RMSE claims require actual ball annotations and cannot be proven from detector confidence logs alone.
- `next build` rewrites `web/next-env.d.ts` from the dev route-types import to the production route-types import in this checkout. Restore the original file after build verification to avoid generated churn.
- PowerShell passes `src/autopan/*.py` literally to `python -m py_compile`; build an explicit file array with `Get-ChildItem` and pass `@files`.
- For this Next app on Windows, start the dev server through `node_modules/.bin/next.cmd dev -H 127.0.0.1 -p <port>` rather than relying on `npm run dev -- --hostname ... --port ...`; the npm forwarding path can turn flags into positional project-directory arguments.
- On GPU VMs with a working system PyTorch/CUDA stack, do not let `pip install ultralytics` replace Torch in a fresh venv. Use a `--system-site-packages` venv, install ordinary dependencies normally, then install `ultralytics` with `--no-deps` so the driver-compatible Torch wheel stays in use.
- Pass an absolute `project` path into Ultralytics training/eval. With a relative project path, Ultralytics can prepend its configured `runs_dir` and save under paths like `runs/detect/runs/ball_v5/...`.
- Keep fetched remote YOLO metadata separate from local prepared datasets. A `data.yaml` copied from a VM can point at inaccessible remote paths and make local verification look broken even when the ZIP is available locally.
- Do not rely on `py_compile` alone for Python pipeline verification. It does not import native/video dependencies, so run CLI/import smoke checks for `cv2`, `py360convert`, pandas, Ultralytics, and the autopan entry points.
- If Ultralytics logs `EMA contains NaN/Inf`, keep the active job running for evidence but prepare a follow-up run with `--no-amp`, a lower `--lr0`, and explicit warmup/weight-decay controls rather than assuming the latest checkpoint is healthy.
- When launching remote background jobs through PowerShell/SSH, avoid complex nested `$` escaping in a one-liner. Upload a reusable shell script first, then run a minimal `nohup bash script > log 2>&1 &` command and immediately verify the pidfile/log path.
- Roboflow YOLO exports may mix detection rows and segmentation polygon rows. Normalize all labels to single-class detection boxes and remove stale `*.cache` files before training so Ultralytics does not silently strip mixed segments at runtime.
- On Windows, `subprocess.run(["npm", ...])` may fail with `[WinError 2]` because the executable is `npm.cmd`. Resolve command paths with `shutil.which` and `.cmd`/`.exe` fallbacks in verifier scripts.
- In PowerShell, `$PID` is a built-in read-only variable. Use names like `$evalPidValue` for process polling scripts so status checks do not accidentally inspect the shell process instead of the background job.
- If a remote training process is orphaned with `PPID=1`, do not assume the original shell wrapper will run post-training evaluation/promotion. Launch a separate logged finalizer that waits for the training PID and performs the final copy/eval/promotion steps.
- Keep local and remote YOLO metadata normalized after any `--no-merge-ball-classes` experiment. A stale `data.yaml` with `nc: 2` can coexist with labels that are all class `0`, making local eval less comparable to the remote one-class training run.
- Keep remote artifact state and local cached artifact state explicit in status output. A stale local `models/ball_v5.pt` or `results/ball_v5_eval.json` can coexist with an unfinished remote run and should not be mistaken for the current training outcome.
- Interpolate longitude tracks in an unwrapped coordinate space. Raw interpolation between `+179` and `-179` crosses through zero and corrupts ball-GT RMSE/recall around the equirectangular seam.
- Unwrap Insta360 Studio pan keyframes before interpolation for the same reason; otherwise the ground-truth curve can take the long path across the panorama seam.
- Aggregate per-clip evaluation metrics by frame count, not by unweighted segment count. Otherwise short segments can skew RMSE/MAE and make approach comparisons misleading.
- Remote wait loops should treat zombie PIDs as exited. Plain `ps -p` can remain true for defunct processes, keeping queued finalizer/follow-up scripts asleep after the useful process has ended.
- Do not overwrite a shell script file while a background bash process is executing that same file. Bash may resume reading from a changed/truncated file offset and fail with parse errors. Upload script revisions to a new filename, or wait until the running process exits.
- Avoid inline SSH one-liners for launching background training jobs from PowerShell. Put the launch logic in a script file, upload it, then run `bash script.sh` so `$ts`, log paths, and PID files are created on the remote host predictably.
- Live training sync should name the active Ultralytics run and write a manifest. Generic files like `ball_v5_live_best.pt` are convenient caches, but they are ambiguous once multiple follow-up runs exist.
- When preserving intermediate checkpoints in git, enable Git LFS first and force-add only the intended ignored `*.pt` files. Keep live follow-up cache checkpoints out of the commit unless explicitly requested, because they can be large and ambiguous.
- Do not convert detector-derived training/predictor CSVs into "ground truth" just to unblock metric gates. They can be useful fixtures or pseudo-label evidence, but recall/RMSE gates need independently annotated ball positions or clearly labeled pseudo-metrics.
