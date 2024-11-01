# proclib
Multi-purpose processing library for downstream use



designed usage:

```python
import unibox as ub
from transformers import AutoModel

# Load models
hands_model = AutoModel.from_pretrained("arot/hands_classifier_v5")
processor = AutoProcessor.from_pretrained("arot/hands_classifier_v5")

# Load image
image = ub.loads("path_to_image.jpg")
inputs = processor(images=image, return_tensors="pt")

# Perform inference
with torch.no_grad():
    logits = hands_model(pixel_values=inputs['pixel_values'])

# Apply the custom postprocessing defined in the model class
results = hands_model.postprocess(logits)

print(results)  # Dictionary of predictions and probabilities
```
