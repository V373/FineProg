# Composite (Weighted-Sum) Encoder Loss Module
# Provides CompositeEncoderLoss: orchestrates multiple BaseEncoderLoss instances
# and returns a weighted-sum scalar, transparent to train.py.
# Use build_loss("composite", config_path=...) to instantiate.
