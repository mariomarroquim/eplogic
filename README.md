EpLogic
=======

Scripts for Anti-HLA antibody target prediction via machine learning.

Description
-----------
These datasets comprise mismatched eplets from HLA alleles tested in single antigen panels of patients waiting for a solid organ transplantation. The experiments described here have the aim of classifying each eplet/panel/patient as reactive or non-reactive. Tip: reactive eplets usually have greater MFI values.


Experiments
-----------

* 1 - Uses the PE dataset for training and the SP dataset for validation. The PE dataset was balanced (under-sampling) before the training with the Random Forest algorithm. The model yielded 89% accuracy and 88% AUC-ROC .
* 2 - Uses the PE and SP datasets for cross-validation. Both were balanced (under-sampling) before the cross-validation with the Random Forest algorithm. The model yielded 82%/88% mean accuracy and 88%/92% mean AUC-ROC for the PE/SP datasets, respectively.

Support
-------

You can contact me at mariomarroquim@gmail.com.
