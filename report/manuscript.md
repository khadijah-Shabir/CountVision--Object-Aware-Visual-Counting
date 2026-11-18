# Object-aware image division for visual counting
A reproducibility study of LVLM-Count with Qwen2-VL-7B

## Abstract
Large vision-language models can miscount crowded scenes. We reimplement LVLM-Count, a training-free wrapper that locates target objects, divides the image along paths that avoid detected masks, and sums counts from the resulting regions. With Qwen2-VL-7B and a documented answer plausibility rule, paired estimates favor the wrapper over whole-image counting on FSC-147, Pascal, Emoji, and the 96 complete Penguin images. The wrapper raises FSC-147 RMSE, and simpler variants achieve lower MAE on FSC-147 and Penguin. The evaluation also documents extreme numeric outputs, incomplete preprocessing, and dependence on the benchmark and scoring policy.

## 1. Introduction and related work
Visual counting requires identifying the requested category and distinguishing individual instances, often under occlusion or at small spatial scales. Direct vision-language counting combines both tasks in one prompt. LVLM-Count [1] proposes counting smaller regions and using detection and segmentation to keep region boundaries from cutting across targets. We study the improvement over direct counting and ask whether the full method also improves on simpler divisions.

Supervised counting benchmarks such as FSC-147 [2] have led to exemplar-based and text-prompted counters. Our comparison concerns a frozen vision-language backbone [3] and its inference-time variants. Grounding DINO [4] localizes a text phrase and SAM [5] provides spatial masks. The deterministic mask-count diagnostic tests a different use of the same detection stack, namely counting surviving masks directly.

## 2. Method
1. Target selection. The benchmark manifest supplies the target phrase and question. FSC-147 uses its annotated class name. Free-form expression extraction is part of the conceptual pipeline, but the evaluated runs use supplied phrases.
2. Area proposal. Grounding DINO Swin-B processes the image with box and text thresholds of 0.1. Overlapping boxes are merged into crop areas. An empty detection list falls back to the complete image.
3. Segmentation. Each area is detected again using the singularized phrase. A largest-first group-box filter removes a box if it covers more than 67% of the bounding rectangle of the smaller boxes. The remaining boxes prompt SAM ViT-H. Mask NMS uses IoU 0.4, followed by 2-pixel erosion, 3-pixel closing, and removal of masks exceeding both half the crop area and 25 times the median mask area.
4. Division. Five pixels per retained mask are projected onto the horizontal axis. Mean shift with the scikit-learn default bandwidth groups these coordinates. Adjacent groups propose a cut midway between their facing extremes. A* searches top to bottom on the free pixels, with an eight-connected grid, Manhattan heuristic, and forbidden diagonal corner cutting. The heuristic can overestimate distance with unit diagonal steps. Coarse cells are traversable only if their underlying pixels are free; finer-grid retries recover paths unavailable at coarse resolution. A failed cut is omitted.
5. Counting. Each resulting region retains its original pixels and has white fill elsewhere. Regions smaller than 100 pixels contribute zero without an inference call. Qwen receives the original question with zero-aware and output-format instructions, and counts are summed over regions and proposal areas.

The mask and group filters affect what can be cut. Spurious masks can block a cut, and missed objects can be sliced or excluded from a proposal area. Detection masks guide division in the full method. Only the explicit mask-count diagnostic uses their number as the prediction.

## 3. Experimental design
FSC-147 uses 1,190 test images with point-count ground truth. Pascal uses the original authors’ 102 VOC 2007 counting questions [1,6], including five zero cases. Only 43 released labels agree with the corresponding raw VOC count when difficult objects are included. There was no independent visual adjudication. Emoji contains 415 questions over 82 images with target counts from 30 to 50. Penguin uses 100 images sampled with inverse count-frequency weighting from labeled separate-site test records [7]; the observed sample spans 0–196. Synthetic Circles contains 220 generated images: 20 at each of 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, and 200 objects.

The active runs use Qwen2-VL-7B-Instruct in fp16 with its shipped near-greedy generation configuration. Base and full use six trials where both exist. The ablations and Synthetic Circles baseline use three. Base queries the entire image. Naive variants use two, three, or four equal vertical strips. Area-naive variants first detect proposal areas. Area-only counts each proposal crop; no-area applies object-aware division to the original image; SAM-all uses automatic masks without phrase-specific mask prompting. The deterministic diagnostic sums surviving masks across crops.

Answers are parsed with a strict double-bracket format first. Lenient parsing accepts a standalone number or number word, including single-bracket forms, and unresolved jobs are retried up to ten attempts. A single parsed answer above 10,000 contributes zero under the main scoring policy. This threshold was adopted after extreme test outputs were observed, applies equally across variants, and retains raw sums for sensitivity analysis. It is a post hoc validity rule.

MAE, RMSE, median absolute error, and tolerance accuracy are calculated by trial and averaged. Paired estimates instead average predictions per question across trials before taking absolute errors. We resample source images 10,000 times with seed 1234, keeping all questions for a sampled Emoji image together. Percentile intervals describe the change in MAE on the observed benchmark population under this resampling scheme. Every Penguin variant comparison uses the same 96 available images. The per-variant coverage and scores on each variant’s available records are supplied separately.

## 4. Results
The paired intervals favor full over base on the four evaluated comparisons. The effect is conditional on the chosen backbone, available records, and scoring policy. On FSC-147, full improves MAE and median error but worsens RMSE, so the squared-error metric does not support a broad reduction in severe misses. The filtered MAE of no-area and area-naive-3 is slightly lower than full. On the Penguin common subset, naive-4 and no-area also achieve lower MAE than full. These comparisons do not isolate a universal benefit from mask-aware paths or the area-detection stage.

On Pascal, exact-match accuracy rises from 24.51% to 30.23%. On Emoji, accuracy within five rises from 4.50% to 13.29%. Synthetic Circles has only a whole-image baseline, with MAE 16.51 and median absolute error 1.00. This gap between mean and median motivates reporting the error distribution; it cannot measure the benefit of division.

## 5. Failure analysis and coverage
Four Penguin inputs failed during full-method preprocessing, and two of those also failed for no-area. Logs attribute the failures to CUDA allocation errors. All six full trials share the same 96 records. Full mask-count cache coverage is also 96. Complete-case comparisons control the denominator, but performance on the omitted inputs remains unknown.

FSC-147 has 22 flagged answers across variants, including six for base and none for full. One repeated base answer is 659,135,152. Base unfiltered MAE exceeds 553,000, compared with 35.75 after the plausibility rule. This raw metric is dominated by extreme replies and the filtered metric depends on the scoring rule. The raw and filtered views are reported together below.

Strict-format failures are common even when lenient parsing recovers the count. The worker-level table counts jobs with one or more strict failures and jobs that exhausted parsing attempts. These counts describe formatting and parser behavior, and do not measure semantic accuracy.

A 30-image FSC-147 validation diagnostic found that 15.6% of requested cuts could not be drawn with the final mask post-processing, and three images had no successful cuts. The original study also used a deliberately selected strawberry example to inspect a successful partition. The public package omits benchmark imagery.

## 6. Limitations and conclusion
The study covers one open backbone and near-greedy decoding. The available artifacts do not establish results for the original paper’s backbones, the planned Qwen2.5 runs, TallyQA-Complex, a finalized higher-temperature sensitivity experiment, or rerun specialist counters. There has been no independent Pascal annotation audit. Penguin inference on the missing records and a full Synthetic Circles comparison remain open. The wrapper also makes multiple inference calls per question, and the saved records lack comparable latency measurements.

The evaluated wrapper improves average count error over direct Qwen2-VL-7B counting under the documented protocol. Its advantage over simpler variants is dataset-dependent, and the effects of answer filtering and missing coverage remain essential to interpreting the reported results.

## References
[1] M. F. Qharabagh, M. Ghofrani, and K. Fountoulakis. LVLM-Count: Enhancing the Counting Ability of Large Vision-Language Models. arXiv:2412.00686, 2024.
[2] V. Ranjan et al. Learning To Count Everything. CVPR, 2021.
[3] P. Wang et al. Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution. arXiv:2409.12191, 2024.
[4] S. Liu et al. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. ECCV, 2024.
[5] A. Kirillov et al. Segment Anything. ICCV, 2023.
[6] M. Everingham et al. The PASCAL Visual Object Classes (VOC) Challenge. International Journal of Computer Vision, 2010.
[7] C. Arteta, V. Lempitsky, and A. Zisserman. Counting in the Wild. ECCV, 2016.
