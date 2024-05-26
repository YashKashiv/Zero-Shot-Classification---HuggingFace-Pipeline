import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
from pylab import rcParams

rcParams["figure.figsize"] = 10, 5

zero_shot_classifier = pipeline("zero-shot-classification")

sequence = "Can you help me to book a cab and order pasta for me"
candidate_labels = ["Flight Travel", "Cabs Travel", "Food", "Movies"]

result = zero_shot_classifier(sequences=sequence, candidate_labels=candidate_labels, multi_class=True)

plt.bar(result["labels"], result["scores"])
plt.yticks(list(np.arange(0, 1.1, 0.1)))
plt.xlabel("Labels")
plt.ylabel("Scores")
plt.title("Zero-Shot Classification Results")
plt.show()