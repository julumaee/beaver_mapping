# Roadmap v4: RF Accuracy & GUI Simplification

Proposal written 2026-09-19 after reviewing `spectral.py`, `training_data.py`, `polygonizer.py`, `models/random_forest.py`, `models/evaluate.py` and `app.py`, and after checking the real data in `data/`.

Effort: **S** ≈ under half a day, **M** ≈ 1–2 days, **L** ≈ several days or needs new data. Priority: **P1** do first, **P2** next, **P3** later or optional.

---

## Baseline: what the data and code look like today

- **Labels** (`data/labels/feature_map(1).kml`): 49 `dead_forest`, 50 `beaver_flood`, 85 `hard_negatives`, 85 `dam` (excluded), 11 `other` (excluded). That makes **~99 independent positive locations**, which is the real sample size.
- **The chips and model are stale.** `data/chips/manifest.csv` still has an older label set: `wet_forest` 36, `beaver_flood` 36, `possible_dam` 2 and 96 negatives. It also has 588 positive chips against 99 negative chips, because positives are augmented ×7 and negatives are not.
- **We don't currently know the model's accuracy.** Spatial CV forms 131 clusters, and only 5 of them contain both classes. For the 126 single-class folds, recall or precision is computed as 0 (`evaluate.py:59-66`) and then averaged in (`evaluate.py:170-174`). The reported mean metrics are therefore meaningless. Phase 0 has to come first, because every other change needs a trustworthy number to judge it.
- **The deferred rotation/flip augmentation idea won't help the RF.** Every RF feature is rotation-invariant: band statistics, NDVI/NDWI statistics, gradient std, GLCM averaged over 0° and 90°, and connected components. Rotated chips would produce near-duplicate feature vectors, so the idea only applies to the CNN.

---

## Phase 0: Trustworthy measurement (P1)

- [x] **R0.1: Pooled out-of-fold spatial CV.** Replace per-fold metric averaging with pooled out-of-fold (OOF) probabilities.
  - Use `GroupKFold` (k=5–10) over the spatial clusters instead of 131 leave-one-cluster-out folds, which is also much faster.
  - Report PR-AUC, ROC-AUC, and precision/recall/F1 at the chosen threshold.
  - Add a **recommended threshold** (max F1, or the best precision at recall ≥ 0.8).
  - Save the OOF predictions to `chips/oof.csv`. R1.5, R3.5 and the GUI confusion matrix all reuse it. *(S)*
- [x] **R0.2: Per-class evaluation from the actual label types.** `evaluate_rf_per_class` hardcodes `wet_forest`/`beaver_flood`/`negative` (`evaluate.py:248`), so `dead_forest`, `flood` and `hard_negatives` never appear. Report hand-labelled hard negatives separately from auto-sampled negatives, since the hard negatives are the ones that matter. *(S)*
- [x] **R0.3: Remove in-sample metrics.**
  - "Evaluate RF vs CNN" scores the *saved* models on a random subset of the chips they were trained on (`evaluate.py:25-55`), and augmented copies of the same point leak across the split. Retrain both models inside spatial folds, or label the tab clearly as indicative only.
  - The GUI confusion matrix uses the final model on its own training data (`app.py:800-820`). Build it from R0.1's OOF predictions instead. *(S)*
- [ ] **R0.4: Detection-level benchmark.**
  - Fully annotate 2–3 tiles that are *not* used for training: digitise every flood and dead-forest area as polygons.
  - Run the full `detect` pipeline on them and measure polygon-level precision/recall.
  - Chip CV only measures chips centred on labels. The real detector scans a 64 px grid everywhere, and only this benchmark reflects what you see in Google Earth. *(M, mostly labelling time)*
- [ ] **R0.5: Rebuild the chips and model from the current labels**, with chips kept in `data/chips/`. Record the R0.1/R0.4 numbers as the baseline for everything below. *(S)*

## Phase 1: Training data quality (P1, the largest expected gain)

With ~99 positive locations, better data will help more than a better model.

- [x] **R1.1: Turn polygon labels into multiple samples.** Polygon placemarks currently collapse to one centroid (`training_data.py:123-127`). Sample area-weighted points inside each polygon: a large flood then yields many chips, and a crescent-shaped flood no longer yields a centroid that sits on dry land. This encourages digitising floods as polygons rather than points. *(S)*
- [ ] **R1.2: Hard-negative mining loop.**
  - Run detection on the training tiles and take high-probability patches that are ≥ 200 m from any positive label.
  - Add them as `mined_negative` and retrain; repeat 2–3 rounds.
  - Why: random negatives are mostly easy (plain forest, fields) and teach the model little.
  - Risk: unlabelled real beaver sites get mined as negatives, so pair this with the review queue (G2.1). *(M)*
- [x] **R1.3: Stop treating unknown label names as positive.** `FEATURE_TO_LABEL.get(ftype, 1)` (`training_data.py:164, 333`) turned `possible_dam` into 14 positive chips, and any mistyped folder name silently becomes a positive. Exclude unknown names and print a warning listing them. *(S)*
- [x] **R1.4: More and more diverse negatives.**
  - Sample 3–5× as many negative locations as positive *chips*: half from the stream corridor, half from anywhere in the imagery.
  - Consider offset-augmenting hand-labelled hard negatives, the same way positives are augmented.
  - Today there are 99 negative chips against 588 positive chips. `class_weight` balances the weights but can't add negative diversity. *(S)*
- [x] **R1.5: Label audit.**
  - From the OOF predictions, list labels the model strongly disagrees with: positives with p < 0.2, negatives with p > 0.8.
  - These are typically misplaced or wrong points. Show them as a layer on the Map. *(S, after R0.1)*
- [ ] **R1.6: Use the dam points.** 85 dams are labelled but ignored. Impoundments sit directly upstream of dams, so use dams to *propose* candidate flood areas for review, not as automatic positives. *(M, optional, P3)*

## Phase 2: New features (P2)

- [ ] **R2.1: Terrain from the MML 2 m DEM (korkeusmalli 2 m, open data).**
  - Features: slope, local relief/TPI, topographic wetness index, height above nearest drainage.
  - Beaver ponds form on flat, low-gradient valley bottoms. This separates them from ditches on slopes, upland clearings and hilltop bogs.
  - The DEM is already in EPSG:3067, so only a windowed read is needed. Likely the single most useful new data source. *(M–L)*
- [ ] **R2.2: Stream context features from Virtavesi (already loaded for masking).**
  - Features: distance to the nearest stream line, stream class/width, and the number of stream segments in the patch.
  - Beaver floods sit on small streams, and a feature captures that better than a hard 100 m cut. *(S–M)*
- [ ] **R2.3: Mapped permanent water.** Add the fraction of the patch covered by mapped lakes and river areas from the MML topographic database. Permanent lakes have the same high NDWI as beaver floods and are a likely false-positive source. Use it as a feature or as a post-filter. *(S)*
- [x] **R2.4: Dead-forest spectral cues.**
  - In CIR imagery, dead trees look grey/cyan: low NIR relative to Red+Green, low saturation, high texture.
  - Add an NIR/(R+G) ratio, CIR saturation `(max−min)/max`, and the fraction of low-NDVI pixels inside high-texture areas.
  - Compute GLCM on NDVI as well as NIR, and consider LBP histograms. *(S)*
- [x] **R2.5: Feature pruning.** 99 correlated features on ~100 independent locations invites overfitting. Use spatial-CV permutation importance and drop near-zero features; this also speeds up inference. *(S)*
- [ ] **R2.6: Multi-temporal change.** Beaver activity shows up as *new* water or dieback where earlier imagery shows living forest. Compare with older MML orthophotos or Sentinel-2. This is the strongest signal in the literature but also the most work: co-registration and radiometric normalisation. *(L, P3)*

## Phase 3: Model (P2)

- [x] **R3.1: Regularise the RF.** It currently uses the defaults: 100 trees, unlimited depth, `min_samples_leaf=1` (`random_forest.py:33`). Grid-search `min_samples_leaf` (3–10), `max_features` and 300–500 trees under grouped CV. *(S)*
- [x] **R3.2: Try `HistGradientBoostingClassifier` and `ExtraTrees`** under the same CV, and keep whichever wins on PR-AUC. *(S)*
- [ ] **R3.3: Multi-class output.** Train on `dead_forest` / `flood` / `negative` and compute P(beaver) as the sum of the two positive classes. Water and dead forest are spectrally opposite, so one binary class forces the model to learn two unrelated appearances. Per-type output also lets the KML say *what* was found. *(S–M)*
- [x] **R3.4: Probability calibration** using `CalibratedClassifierCV` on grouped folds, so that "0.7" really means ~70 %. *(S)*
- [x] **R3.5: Model metadata sidecar** (`model.json`) containing the label mapping, feature-vector version/length, recommended threshold, CV metrics, training date and chip counts. Detect and the GUI read the threshold from it. It also catches old models whose feature length no longer matches. *(S)*

## Phase 4: Detection post-processing (P1–P2)

- [x] **R4.1: Make the minimum-area filter effective.** One patch is 32 × 32 m = 1024 m², which already exceeds `MIN_AREA_M2 = 500` (`polygonizer.py:14`), so single isolated patches always pass. Require ≥ 2 connected patches, or raise the limit and make it configurable. *(S, P1)*
- [x] **R4.2: Smooth and use hysteresis thresholding.** Apply a 3 × 3 mean filter to the probability map, then seed regions at ≥ 0.7 and grow them at ≥ 0.5. This removes speckle false positives while keeping flood outlines. *(S, P1)*
- [ ] **R4.3: Wider or softer stream mask.** Large floods can extend more than 100 m from the stream line and get clipped at the mask edge. Widen the mask to 200–300 m and let R2.2's distance feature do the fine discrimination. *(S)*
- [ ] **R4.4: Optional 32 px stride** for finer outlines (4× the compute). *(M, P3)*
- [ ] **R4.5: Stop loading the full tile into memory.** `detect_rois_rf_segmentation` reads the whole tile into RAM and pads it (`polygonizer.py:83-93`), against the project's memory constraint. Read in row strips with a 512 px halo instead. This doesn't affect accuracy. *(M, P3)*

**Suggested order:** R0.1–R0.5 → R1.3, R4.1, R4.2 (quick wins) → R1.1, R1.4, R1.5 → R3.1, R3.5 → R1.2 with G2.1 → R2.1, R2.2, R2.3 → the rest.

---

## GUI: simple by default, complete when needed

Currently there are 8 tabs. Imagery is entered in 5 of them, labels in 4, hydrography in 4 and the RF model path in 4, adding up to about 40 settings keys that must be kept in sync by hand.

### G0: Fix the Map tab (P1)

**Root cause (reproduced):** with the saved defaults, `handle_load_map` takes **104 s** and returns a **617 MB** HTML string. The browser never renders it.

- The default detections file `data/output/detections.kml` contains 7,220 polygons spanning 138 × 138 km.
- `_build_map` uses that file's extent as the hydrography window (`app.py:634-637`).
- `_add_hydro_layer` (`app.py:540-606`) loads *every* stream and water body in that window from all 10 GPKGs (1.1 GB) into one uncapped GeoJSON layer.
- Everything is embedded inline as the `srcdoc` of an iframe (`app.py:705`).

Measured without hydrography: detections alone give 8.3 MB in 4 s, and labels alone 0.27 MB in 0.4 s.

The 7,220 detections are themselves a symptom of the accuracy problems above (see R4.1/R4.2).

- [x] **G0.1: Cap the hydrography layer.**
  - Load hydrography only within a bounded window, about 10 × 10 km around the centre of the detections or the current view.
  - Drop the small `virtavesikapea` streams and simplify more strongly (20–50 m) when the window is large.
  - Show a notice when the limit applies. *(S)*
- [x] **G0.2: Lighter detection layer.** Draw detections as one GeoJSON layer, round coordinates to 5 decimals, and apply the confidence filter *before* rendering. Don't parse the KML twice per load. *(S)*
- [x] **G0.3: Size guard.** If the generated page exceeds about 30 MB, show a readable message ("too much to draw — zoom in or raise the threshold") instead of sending it. *(S)*
- [x] **G0.4: Fix click-to-diagnose.**
  - The Leaflet click handler runs *inside* the folium iframe and sets `window._mapClickLat/Lon` there (`app.py:692-693`).
  - The Diagnose button reads them from the parent page (`app.py:1489`), so it always sends the fallback 25.0/62.0.
  - Set `window.parent._mapClickLat/Lon` as well, and switch to the Diagnose view after clicking (later replaced by G2.3). *(S)*
- [ ] **G0.5 (later): Serve the map as a file.** Write it to a file and load it in the iframe by URL instead of embedding it inline, and render hydrography as raster tiles. *(M, P3)*

### G1: Simplify (P1)

- [x] **G1.1: One shared "Project" panel** at the top, with imagery, labels and hydrography directories and a project folder, auto-filled from `data/`. Derive `model.pkl`, `model.json`, `chips/`, `manifest.csv`, `norm_stats.json` and timestamped output names from it. This removes ~30 duplicated fields. *(M)*
- [x] **G1.2: Workflow-ordered tabs:**
  1. **Data check:** the current Overview plus pre-run validation.
  2. **Train:** RF, with a "run spatial CV afterwards" checkbox that is on by default.
  3. **Evaluate:** spatial CV, the OOF confusion matrix, the PR curve and the label audit.
  4. **Detect:** the threshold defaults to the model's recommended value.
  5. **Map & Review:** detections, labels and hydrography, with inline Diagnose and the review queue.
  6. **Experimental (CNN):** Train CNN plus RF vs CNN, collapsed or hidden behind a toggle. *(S–M)*
- [x] **G1.3: Put advanced knobs in "Advanced" accordions with `info=` help text.** These are augmentation count, tulvaalue flood samples, the hydro-negatives setting, cluster radius, test fraction, epochs and learning rate. Rename jargon, e.g. "Extra training samples from mapped flood areas (MML tulvaalue)". *(S)*
- [x] **G1.4: Auto-save settings on change** and drop the "Save as defaults" button. Today the augment slider, basemap and layer toggles are never saved. *(S)*
- [x] **G1.5: Always keep chips** in the project `chips/` folder, so Evaluate works without remembering to fill in "Chip directory" during training (`cli.py:82-86`). *(S)*
- [x] **G1.6: Remove the `pip install` on every launch** (`app.py:6-12`). It is slow and fails offline. Check imports instead and print a one-line hint if something is missing. *(S)*
- [x] **G1.7: Use the current label names.** Build the Map legend, colours and the per-class checkbox from the label types actually found; they still use `wet_forest`/`beaver_flood`. Make Overview count by KML folder the same way training does. *(S)*

### G2: Make the GUI the accuracy loop (P2)

- [ ] **G2.1: Review queue (active learning).**
  - On the Map & Review tab, step through detections sorted by uncertainty. Show the chip, NDWI and the RF probability map.
  - Buttons: **Beaver / Not beaver / Skip**. Answers are appended to `labels/review.kml` in the correct folders, and the next training run picks them up automatically.
  - This turns the verification work you already do into training data (pairs with R1.2). *(M–L)*
- [x] **G2.2: Label-audit layer** on the map from R1.5: suspicious labels shown as red rings you can click into Diagnose. *(S)*
- [x] **G2.3: Merge Diagnose into Map.** Clicking the map runs the diagnosis in a side panel, instead of silently filling lon/lat in another tab that defaults to 25.0/62.0. *(M)*
- [x] **G2.4: Pre-run validation** in Data check and before each run:
  - paths exist, and the number of tiles found;
  - label counts by class, plus unknown folder names (R1.3);
  - labels outside tile coverage, and hydrography coverage per tile. *(M)*

### G3: Polish (P3)

- [x] **G3.1: Timestamped detection outputs** plus a "past runs" dropdown on the Map; use one shared threshold between Detect and the Map export filter. *(S)*
- [x] **G3.2: Replace upload-based Browse buttons** (which copy to Gradio temp paths) with dropdowns listing the files in the project's `models/` folder. *(S)*
- [x] **G3.3: Progress bars for training and evaluation**, like Detect already has. *(S)*
- [x] **G3.4: Status line** such as "Model trained 2026-05-27 · 588 pos / 99 neg chips · CV PR-AUC 0.xx · model older than labels ⚠". *(S, needs R3.5)*
