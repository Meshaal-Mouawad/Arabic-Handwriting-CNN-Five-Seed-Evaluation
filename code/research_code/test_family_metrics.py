"""Metric regression: cross-family false positives must affect family macro-F1."""
import numpy as np
from benchmark import eval_metrics

# A digit 0 is correctly classified; a letter 10 is wrongly predicted as digit 0.
# Global class-0 precision is 1/2 and recall is 1, giving F1=2/3.
# The digit-family macro-F1 over ten classes must therefore equal (2/3)/10.
y=np.array([0,10])
p=np.array([0,0])
r=eval_metrics(y,p)
assert abs(r['macro_f1_digits']-(2/3)/10) < 1e-8, r
assert r['macro_f1_letters'] == 0, r
assert r['family_confusion_from_final_prediction'] == [[1,0],[1,0]], r
print('PASS: global cross-family errors are included in macro-F1.')
