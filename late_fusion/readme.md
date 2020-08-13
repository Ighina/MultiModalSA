# Pre-trained Late Fusion model

This folder contains the pre-trained late fusion model, as described in the accompanying paper, as well as the hyperparameters used to train it, included in Hyperparameters.json.

To use the pre-trained model for testing on CMU-MOSEI, from the main folder run:
```
run_test.py -folder late_fusion
```

Note that the model was trained on GPU and, as such, quite different results might be obtained if used with a CPU instead.
