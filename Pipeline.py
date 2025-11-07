# Pipeline.py — shim in case the pickled model references a module named "Pipeline"
from sklearn.pipeline import Pipeline as Pipeline
