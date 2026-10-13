# Bundled data attribution

The bundled datasets are public publisher datasets, not synthetic data. The project code and demonstration reports do not change their ownership.

## BANKING77

Publisher: PolyAI.

Citation: Iñigo Casanueva, Tadas Temcinas, Daniela Gerz, Matthew Henderson, and Ivan Vulić (2020). *Efficient Intent Detection with Dual Sentence Encoders*. Proceedings of the 2nd Workshop on NLP for ConvAI, ACL 2020.

Source: https://github.com/PolyAI-LDN/task-specific-datasets

Paper: https://arxiv.org/abs/2003.04807

Publisher license: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/

Changes: original train/test CSV files concatenated into a canonical CSV; `category` renamed `label`; `split` added. All original texts/labels and split assignments preserved. Separate audit reports describe quality signals; they are not publisher endorsements.

## SMS Spam Collection

Creators: Tiago Almeida and Jos Hidalgo.

Citation: Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84

Publisher source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

License listed by UCI: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/

Changes: original tab-separated source converted to UTF-8 CSV; demonstration split assignments added using a normalized-text hash bucket. Texts and labels preserved. Splits are project-created and are not supplied/endorsed by the creators. Original source ZIP and publisher readme retained.

Exact downloaded bytes and processed bytes are identified in each `provenance.json`. Neither source endorses this auditor or its provisional policy weights.
