# Related work notes (verified 2026-10-15)

Compiled from primary pages (arXiv abstract/HTML, Crossref, ACL Anthology, PMLR, NeurIPS
proceedings, HF model cards, GitHub). DBLP and ACM DL full texts were not reachable (bot check / 403).
BibTeX for the cited entries is in `paper/references.bib`. **U** = unverified.

## Dataset provenance

**MADE-WIC** (Mock et al., ASE 2024 tool demo, pp. 2346-2349, doi 10.1145/3691620.3695348; arXiv 2408.05163):
- Fuses OSPR (Chromium, Linux, Firefox; 688,134 functions, no original vulnerability labels),
  Devign (FFmpeg, Qemu; 27,282) and Big-Vul (10 projects; 144,358). 859,774 functions in total, with >2.7M comments.
- Annotations:
  - **MAT**: "Matches task Annotation Tags" of Guo et al. (TOSEM 2021). A function is technical debt if at least one comment contains TODO, FIXME, XXX or HACK.
  - **PS**: the 64 Potdar & Shihab patterns.
  - **SecI**: security indicators.
  - **W**: CWE-derived weakness heuristics from WeakSATD (Russo et al., MSR 2022).
- Functions and comments were extracted with srcML; near-duplicates removed with PMD-CPD.
- The paper reports low agreement between vf and W (8-45% per project) and between PS and MAT.
- Zenodo has several versions. Which one the competition used is U.

**Organizer curation** (github.com/moritzmock/NLBSE2027-setup, read directly):
- weakness = Devign OR Big-Vul OR W.
- clang-format (LLVM style); drop functions with > 500 cl100k tokens; restore the joint distribution; 70/10/20 split.
- `LeadingComment` is dropped from the released files.

**Vulnerability label quality:**
- Croft et al. (ICSE 2023, pp. 121-133): on manual samples, estimated vulnerable-label accuracy is ~54% for Big-Vul and ~80% for Devign; duplication is ~17% and ~10%.
- PrimeVul (Ding et al., ICSE 2025, pp. 1729-1741): reports low label accuracy and train/test duplication in Big-Vul and Devign, and that random splits inflate scores. A 7B model drops from 68% F1 on BigVul to 3% on PrimeVul.
- The competition split is random-stratified, so similar inflation is plausible (our clone audit: 18% of test functions have a type-2 clone in train).

## Multitask SATD + vulnerability

**Russo, Melegati, Mock** (ICPC 2025, pp. 1-12, doi 10.1109/ICPC66645.2025.00017; arXiv 2501.15934). VulSATD:
- Model: CodeBERT with two heads; input `[CLS] comment [SEP] code`; max 512 tokens; lr 2e-5; batch 16; 10 epochs.
- Labels: MAT as the SATD label, W/Devign/Big-Vul for vulnerability (per their replication code).
- Results: multitask and single-task show "no significant differences". Multitask is ~2x faster than two single-task models. Training took ~29 h on 2xA100 for OSPR.
- Their code includes an option to mask SATD keywords (`FIXME|TODO|HACK|\sXXX\s`).

## Encoders (candidates and context)
- CodeBERT (Feng et al., Findings EMNLP 2020): 12 layers x 768, RoBERTa BPE, MIT. The baseline.
- UniXcoder (Guo et al., ACL 2022, pp. 7212-7225): 12 layers x 768, 51,416-token vocabulary, 1026 positions, Apache-2.0.
- GraphCodeBERT (ICLR 2021); CodeT5+ 110M embedding (EMNLP 2023); CodeSage-small (ICLR 2024).
- ModernBERT (ACL 2025): its own measurements show it slower than BERT on fixed-length inputs; its gains come from unpadding.
- Ettin, NeoBERT: general-text encoders.
- Small general encoders: DistilBERT 6 x 768; TinyBERT 4 x 312; MiniLM 12 x 384.

## Efficiency of code models
- **EarlyBIRD** (Grishina, Hort, Moonen, ESEC/FSE 2023, pp. 895-907): classifiers built on early CodeBERT layers. Using 3 of 12 layers gave +2 accuracy points on Devign and 3.3x faster fine-tuning. Directly supports layer-truncated students.
- **Compressor** (Shi et al., ASE 2022): GA architecture search with a GFLOPs-aware fitness plus distillation; a 3 MB student at 59.4% vs 61.8% Devign accuracy.
- **Avatar** (Shi et al., ICSE-SEIS 2024): up to 184x less energy and 76x faster, at ~1.7% effectiveness loss.
- **d'Aloisio et al.** (SANER 2025): on CodeBERT/Devign, int8/int4 quantization (optimum-quanto) *increased* inference time by 102% and 134%. Distillation reduced CPU time 40% at a 10% MCC cost.

## Distillation, multitask optimization, thresholds
- Distillation: Hinton et al. (2015); DistilBERT; TinyBERT; MiniLM.
- Multitask loss weighting: Kendall et al. (CVPR 2018); GradNorm (ICML 2018); PCGrad (NeurIPS 2020).
- Imbalance losses: focal loss (ICCV 2017); asymmetric loss (ICCV 2021).
- F1 thresholding: Lipton et al. (ECML PKDD 2014) show the F1-optimal threshold of a calibrated classifier is half the optimal F1. This motivates per-task thresholds instead of a fixed 0.5 for a 1.65%-prevalence task.

## Efficient inference
- Variable-length batching: fairseq (NAACL 2019 demos); ByteTransformer (IPDPS 2023).
- Early exit: DeeBERT (ACL 2020); PABEE (NeurIPS 2020).
- Token pruning: PoWER-BERT (ICML 2020); LTP (KDD 2022).
- Quantization: Q8BERT (2019); I-BERT (ICML 2021).

## SATD detection
- Potdar & Shihab (ICSME 2014): 62-64 patterns.
- Guo et al. MAT (TOSEM 2021): on Maldonado's Java data, MAT averages F1 0.722 vs CNN 0.764; tags occur in >60% of SATD comments in 8 of 10 projects.
- Ren et al. CNN (TOSEM 2019); Jitterbug (TSE 2022), which notes that "xxx" is project-dependent.

## Prior NLBSE competitions
See research/competition_intelligence.md, section 6.
