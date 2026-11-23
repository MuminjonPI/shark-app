import gradio as gr
import spaces
import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

weights = hf_hub_download(repo_id="MuminjonPI/shark-phone-detector", filename="best.pt")
model = YOLO(weights)

@spaces.GPU
def shark(image):
    if image is None:
        return None, "Show me a photo first!"
    # Gradio gives RGB; YOLO expects BGR, so flip the colour channels
    res = model.predict(image[..., ::-1], conf=0.25)[0]
    out = image.copy()
    n = len(res.boxes)
    for x1, y1, x2, y2 in res.boxes.xyxy.int().tolist():
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 120, 255), 4)
        cv2.putText(out, "CHOMP!", (x1, max(y1 - 10, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 120, 255), 3)
    if n == 0:
        msg = "No phones found. The shark is still hungry 🦈"
    else:
        msg = f"🦈 CHOMP! The shark ate {n} phone(s). Sharks don't text."
    return out, msg

demo = gr.Interface(
    fn=shark,
    inputs=gr.Image(sources=["webcam", "upload"], type="numpy", label="Your photo"),
    outputs=[gr.Image(label="Shark view"), gr.Textbox(label="Result")],
    title="🦈 Shark vs Phones",
    description="Hold up your phone. The shark eats it.",
)
demo.launch()
