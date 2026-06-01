# Ball v5 Training and Dataset Report

Generated: 2026-06-01

## Executive Summary

The headline detector result does not support assuming good Insta360 performance. The stable saved model reports `mAP50=0.8203`, `mAP50-95=0.4381`, precision `0.8881`, recall `0.7846` on the Roboflow `test` split, but that split is about 80% Veo-style 1920x1080 imagery by image count. Filtered evaluation shows a large domain gap:

| Split/subset | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| valid, Insta360-style | 144 | 83 after duplicate removal | 0.716 | 0.639 | 0.607 | 0.234 |
| valid, Veo-style | 541 | 543 | 0.909 | 0.821 | 0.873 | 0.473 |
| test, Insta360-style | 70 | 39 | 0.803 | 0.564 | 0.566 | 0.240 |
| test, Veo-style | 272 | 272 | 0.896 | 0.816 | 0.857 | 0.468 |

Bottom line: the useful critical metric is the Insta360-specific recall, which is only `0.639` on validation and `0.564` on test for the stable model. The `0.82-0.83 mAP50` number is mostly explained by Veo-style performance and should not be treated as an Insta360 detector quality claim.

## Training Image Size

The stable yolo11s run used `imgsz=1280`.

Evidence:

- `run_ball_v5_stable.sh` trains `MODEL=yolo11s.pt` and hardcodes `--imgsz 1280`.
- The same script evaluates the stable checkpoint with `--split test --imgsz 1280`.
- `train_ball_v5.py` defaults to `--imgsz 1280`.
- The yolo11m follow-up script also uses `--imgsz 1280`.
- A separate high-resolution follow-up script defaults to `IMGSZ=1536`, but that is not the completed stable/candidate result discussed here.

Current remote status from `python sync_ball_v5_artifacts.py --status`:

- Stable artifact exists and failed promotion gate: `mAP50=0.82031755817434`, `mAP50-95=0.4380733215838644`, target `0.90/0.60`, `promoted=False`.
- Current live yolo11m follow-up is still running; latest lightweight sync during this report was epoch 33 with `mAP50=0.79019`, `mAP50-95=0.38370`.

## Dataset Used

They are using `data/ball_v5/data.yaml`, extracted from:

`C:\Workspace\2026\David-Ball-Tracking\ball_detector_merged.v3i.yolov8.zip`

The dataset metadata identifies it as Roboflow `ball_detector_merged`, version 3, exported on 2026-05-21, with 3425 images and no image augmentations in the export.

This is not named `ball_dataset_v5_clean`. The local pipeline does clean/normalize the export into `data/ball_v5`, but I found no evidence that a separate `ball_dataset_v5_clean` artifact is the source of the current trained/evaluated model.

## Validation Composition

I classified images by filename and resolution:

- Insta360-style: `frame_*` and `VID_*`, all 1280x720 in this dataset.
- Veo-style: `Fortuna-*`, `Veo-*`, `Video-from-Veo*`, `video5`, all 1920x1080 in this dataset.

| Split | Total images | Veo-style images | Insta360-style images | Veo-style boxes | Insta360-style boxes | Empty Insta360 labels |
|---|---:|---:|---:|---:|---:|---:|
| train | 2398 | 1424 | 974 | 1438 | 654 | 323 |
| valid | 685 | 541 | 144 | 543 | 84 raw / 83 after duplicate removal | 64 |
| test | 342 | 272 | 70 | 272 | 39 | 32 |

The validation split is 79.0% Veo-style by image count and 86.6% Veo-style by raw ball boxes. The test split is 79.5% Veo-style by image count and 87.5% Veo-style by ball boxes.

That means aggregate mAP50 is dominated by Veo-style imagery. This explains why the all-test stable result of `mAP50=0.8203` coexists with only `0.566` mAP50 on Insta360-style test images.

## Model Metrics

Saved aggregate artifacts:

| Artifact | Split | Precision | Recall | mAP50 | mAP50-95 | Notes |
|---|---|---:|---:|---:|---:|---|
| `results/ball_v5_stable_eval.json` | test | 0.888 | 0.785 | 0.820 | 0.438 | stable yolo11s 1280 |
| `results/ball_v5_yolo11s_1280_candidate_eval.json` | test | 0.888 | 0.785 | 0.820 | 0.438 | same stable checkpoint/fingerprint |
| `results/ball_v5_live_eval.json` | test | 0.899 | 0.765 | 0.823 | 0.440 | live cache, ambiguous while follow-up runs exist |
| `results/ball_v5_eval.json` | val | 0.743 | 0.406 | 0.481 | 0.192 | stale/formal final path; not promoted by gate |

Filtered metrics generated for this report:

- `results/ball_v5_subset_valid_insta360.json`
- `results/ball_v5_subset_valid_veo.json`
- `results/ball_v5_subset_test_insta360.json`
- `results/ball_v5_subset_test_veo.json`

These filtered results are the best answer to the critical Insta360 question: stable-model recall is materially worse on Insta360-style imagery than on Veo-style imagery.

## Annotation Corruption Status

Partially addressed, but not fully clean.

What is addressed:

- `train_ball_v5.py` rewrites labels to one class (`0: ball`), converts segmentation polygon rows into detection boxes, and removes stale YOLO cache files during ZIP preparation.
- `results/ball_v5_dataset_audit.json` reports no missing labels, no orphan labels, no out-of-range boxes, no non-bbox rows, and only class `0` across train/valid/test.
- The audit now reports source-domain composition for each split and flags duplicate label rows.
- My direct label scan found no non-bbox rows, no non-zero class IDs, and no out-of-range normalized boxes.

Remaining issue:

- Ultralytics removed one duplicate label during filtered validation eval:
  `data/ball_v5/valid/images/frame_00875_jpg.rf.261a3ecc1e9988cae483f6c4d8b557cf.jpg`.
- Direct scan confirms one duplicate row in:
  `data/ball_v5/valid/labels/frame_00875_jpg.rf.261a3ecc1e9988cae483f6c4d8b557cf.txt`.

So the earlier mixed-class/mixed-segmentation corruption appears addressed, but the dataset is not perfectly clean until duplicate label rows are removed. The current audit makes that issue visible.

## Concerns Answered

| Concern | Answer |
|---|---|
| What `imgsz` was used? | Stable/current comparable result: `1280`. A 1536 high-res run exists as a follow-up path, not the completed stable result. |
| What dataset are they using? | Roboflow `ball_detector_merged` v3 exported 2026-05-21, extracted to `data/ball_v5`. |
| Is validation mostly Veo? | Yes. Validation is 541/685 images Veo-style and 543/627 raw boxes Veo-style. |
| Does 0.83 mAP50 translate to Insta360? | No. Stable filtered validation mAP50 is 0.607 on Insta360-style versus 0.873 on Veo-style; test is 0.566 versus 0.857. |
| What is Insta360-specific precision/recall? | Validation P/R: 0.716/0.639. Test P/R: 0.803/0.564. |
| Are they using `ball_dataset_v5_clean`? | No direct evidence. The active dataset is `data/ball_v5` from `ball_detector_merged.v3i.yolov8.zip`, with normalization applied locally. |
| Corrupted annotations addressed? | Mixed segments/classes and range issues appear addressed. One duplicate label remains; cleanup is still needed and the audit now catches it. |

## Recommendations

1. Do not promote the detector based on aggregate mAP. Gate promotion on `valid_insta360` or `test_insta360` recall/mAP50.
2. Split reporting by source domain in every training/eval run: Veo-style, Insta360-style, and aggregate. Use `train_ball_v5.py --eval-domains --domain-metrics-json results/ball_v5_domain_eval.json` for repeatable subset metrics.
3. Remove duplicate label rows before the final detector training pass and keep duplicate-row detection in `audit_ball_dataset.py`.
4. If `ball_dataset_v5_clean` is the intended canonical dataset, point training scripts at that exact artifact and record its fingerprint in eval JSON.
5. Add a minimum Insta360 recall gate. A reasonable first gate would be `recall >= 0.75` on an Insta360-only validation/test subset, then raise it once the dataset is expanded.
