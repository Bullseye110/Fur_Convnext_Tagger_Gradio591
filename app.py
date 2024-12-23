import gradio as gr
import huggingface_hub
from PIL import Image
from pathlib import Path
import onnxruntime as rt
import numpy as np
import csv
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

# Model and tag setup
e621_model_path = Path(huggingface_hub.snapshot_download('toynya/Z3D-E621-Convnext'))
e621_model_session = rt.InferenceSession(e621_model_path / 'model.onnx', providers=["CPUExecutionProvider"])
with open(e621_model_path / 'tags-selected.csv', mode='r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    e621_model_tags = [row['name'].strip() for row in csv_reader]

def prepare_image_e621(image: Image.Image, target_size: int):
    image_shape = image.size
    max_dim = max(image_shape)
    pad_left = (max_dim - image_shape[0]) // 2
    pad_top = (max_dim - image_shape[1]) // 2
    padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(image, (pad_left, pad_top))
    if max_dim != target_size:
        padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)
    image_array = np.asarray(padded_image, dtype=np.float32)
    image_array = image_array[:, :, ::-1]
    return np.expand_dims(image_array, axis=0)

def predict_e621(image: Image.Image):
    image_array = prepare_image_e621(image, 448)
    input_name = 'input_1:0'
    output_name = 'predictions_sigmoid'
    result = e621_model_session.run([output_name], {input_name: image_array})
    result = result[0][0]
    scores = {e621_model_tags[i]: float(result[i]) for i in range(len(result))}
    predicted_tags = [tag for tag, score in scores.items() if score > 0.3]
    return ', '.join(predicted_tags).replace("_", " "), scores

def predict_batch_e621(files, progress=gr.Progress(track_tqdm=True)):
    if not files:
        return "No images provided"
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for file in files:
            img = Image.open(file.name).convert('RGB')
            result = predict_e621(img)
            results.append(result[0])
    return "\n\n".join(results)

with gr.Blocks() as demo:
    gr.Markdown("## E621 Tagger (Z3D-E621-Convnext)")
    with gr.Tab("Single Image"):
        with gr.Row():
            image_input = gr.Image(label="Source", type='pil')
            tag_string_output = gr.Textbox(label="Tag String", show_copy_button=True)
            tag_predictions_output = gr.Label(label="Tag Predictions", num_top_classes=100)
        submit_btn = gr.Button("Predict")
        submit_btn.click(predict_e621, inputs=[image_input], outputs=[tag_string_output, tag_predictions_output])

    with gr.Tab("Batch Processing"):
        with gr.Row():
            files_input = gr.File(label="Upload Images", file_types=["image"], file_count="multiple")
            batch_results_output = gr.Textbox(label="Batch Results", show_copy_button=True, lines=10)
        batch_btn = gr.Button("Process Batch")
        batch_btn.click(predict_batch_e621, inputs=[files_input], outputs=[batch_results_output])

if __name__ == "__main__":
    demo.launch(share=True)
