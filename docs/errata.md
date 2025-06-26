---
title: Errata
has_children: false
nav_order: 3
---

# Errata

* **06/21/2025:**
    - In cell 20 in `chap09_classification.ipynb` changed this line: `max_tokens = dev_df['tokens'].map(len).max()` to `max_tokens = test_df['tokens'].map(len).max()`.
    - Changed to the `evaluate` library for the `sacrebleu` evaluation metric in cell 8 in `chap15_translation_en_to_ro.ipynb`, cell 8 in `chap15_translation_ro_to_en.ipynb`, cell 9 in `chap15_translation_ro_to_en_finetune.ipynb`, and cell 8 in `chap15_translation_ro_to_en_finetuned.ipynb`.
    - In cell 10 in `chap15_translation_ro_to_en_finetune.ipynb` changed the parameter name `evaluation_strategy` to `eval_strategy`.
    - Fixed incorrect page numbers in the Firth reference.
    
* **04/05/2025:**
    - Corrected the implementation of `remove_diacritics()` in Appendix B, which should contain `if not unicodedata.combining(c)` rather than `if unicodedata.combining(c)`.
    - Corrected the book URL in the first paragraph of Chapter 4 to use `https` instead of `http`.
    - Fixed the fact that the number of epochs parameter (`n_epochs`) was defined but never used in several code blocks in Chapter 4.

* **12/15/2024:** 
    - Corrected equations 16.1 and 16.2.

# Acknowledgements

We thank the following people for discovering these mistakes: Mike Maxwell, Minglai Yang.
