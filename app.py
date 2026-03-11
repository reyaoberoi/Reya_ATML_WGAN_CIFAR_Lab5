import gradio as gr
import torch
import numpy as np
from model import Generator

device = "cpu"

# Load model
model = Generator()
model.load_state_dict(torch.load("models/generator.pth", map_location=device))
model.eval()


def generate_image():

    noise = torch.randn(1,100)

    with torch.no_grad():
        img = model(noise)[0]

    img = (img + 1) / 2
    img = img.permute(1,2,0).numpy()

    return img


with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="orange"),
    css="""
    .container {max-width: 600px; margin: auto;}
    """
) as demo:

    with gr.Column(elem_classes="container"):

        gr.Markdown(
        """
        # 🎨 WGAN CIFAR Image Generator
        Generate synthetic CIFAR images using a trained **WGAN model**
        """
        )

        output = gr.Image(
            label="Generated Image",
            width=320,
            height=320
        )

        with gr.Row():
            generate_btn = gr.Button("Generate", variant="primary")
            clear_btn = gr.Button("Clear")

        generate_btn.click(
            fn=generate_image,
            outputs=output
        )

        clear_btn.click(
            fn=lambda: None,
            outputs=output
        )

demo.launch()