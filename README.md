EpLogic
=======

Scripts for Anti-HLA antibody target prediction via machine learning.

Description
-----------
These datasets comprise mismatched eplets from HLA alleles tested in single antigen panels of patients waiting for a solid organ transplantation. The experiments described here have the aim of classifying each eplet/panel/patient as reactive or non-reactive. Tip: reactive eplets usually have greater MFI values.


Experiments
-----------

* 1 - Uses the PE dataset for training and the SP dataset for validation. The PE dataset was balanced (under-sampling) before the training process with the Random Forest algorithm. The validation was performed without cross-validation and showed an overall accuracy of 88% and an AUC-ROC of 89%.

Support
-------

You can contact me at mariomarroquim@gmail.com.
