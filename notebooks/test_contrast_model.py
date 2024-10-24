from contrast_model.configuration_contrast import ContrastConfig
from contrast_model.modeling_contrast import ContrastModel

# Create a dummy config and model
config = ContrastConfig(image_size=768)
model = ContrastModel(config)

# Manually set torch_dtype for saving compatibility
model.config.torch_dtype = "torch.float32"  # You can set this as needed

# Run the model on a sample image (provide your image path here)
image_path = "/home/ubuntu/dev/proclib/notebooks/sample_image.jpg"
output = model.forward(image_path)
print(output)  # Prints the contrast value

# Save the config and model (this will also save the model code)
config.save_pretrained("contrast_model_hf")
model.save_pretrained("contrast_model_hf")

print("done")