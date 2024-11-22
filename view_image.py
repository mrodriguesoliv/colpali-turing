import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_dataset

ds = load_dataset("mrodriguesoliv/colpali-turing")

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

for i in range(10):
    sample_image = ds["train"][i]["image"]

    axes[i].imshow(sample_image)
    axes[i].axis('off')

plt.show()