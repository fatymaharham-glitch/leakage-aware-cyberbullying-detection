# P5 contextual baseline decision

Frozen `sentence-transformers/all-MiniLM-L6-v2` embeddings at revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf` were evaluated on grouped folds [1, 2, 3, 4]. Fold 0 was not embedded or evaluated.

Best contextual candidate: **linear_svm_c0_5**, mean macro-F1 **0.8270 ± 0.0076**.

The best sparse candidate scored 0.8703; the contextual result was -0.0433 lower. Mean fold fit time was 3.38s after a one-time 54.37s embedding pass, versus 22.26s for the sparse winner. Sparse linear features also offer more direct coefficient-based explanations.

This is a transfer baseline using a frozen general-purpose English encoder, not fine-tuning. Inputs longer than 256 wordpieces are truncated by the selected model.
