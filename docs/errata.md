---
title: Errata
has_children: false
nav_order: 3
---

# Errata

* **04/05/2025:**
    - Corrected the implementation of `remove_diacritics()` in Appendix B, which should contain `if not unicodedata.combining(c)` rather than `if unicodedata.combining(c)`.
    - Corrected the book URL in the first paragraph of Chapter 4 to use `https` instead of `http`.

* **12/15/2024:** 
    - Corrected equations 16.1 and 16.2.

# Acknowledgements

We thank the following people for discovering these mistakes: Michael Lewis, Minglai Yang.
