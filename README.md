# 🦈 Shark vs Phones

**ML Project 1** · Muminjon Khayitboev · November 2026

A webcam game where a shark "eats" any phone you hold up to the camera. A custom-trained object detector (YOLO26n) finds phones in real time, directly in the browser.

| | Link |
|---|---|
| Live app (Vercel) | https://shark-app-web.vercel.app/ |
| Model (Hugging Face) | https://huggingface.co/MuminjonPI/shark-phone-detector |
| Demo app (Hugging Face Space) | https://huggingface.co/spaces/MuminjonPI/shark-app |
| Training notebook | [`notebook/train.ipynb`](notebook/train.ipynb) |

---

## 1. Goal

Inspired by the webcam game *You Are Fish* (you-are-fish.vercel.app), which detects when a player holds up a phone, the aim was to build the full machine-learning pipeline myself:

1. Prepare a dataset
2. Fine-tune an object detector in Google Colab
3. Publish the trained weights on Hugging Face
4. Build a web app that uses the model, deployed on Vercel

## 2. Pipeline

```
COCO 2017 (cell phone)  →  Colab: fine-tune YOLO26n  →  best.pt / best.onnx
        →  Hugging Face model repo  →  Vercel web app (runs in browser)
                                    →  Hugging Face Space (Gradio demo)
```

## 3. Data

- **Source:** COCO 2017, class `cell phone`, downloaded with FiftyOne.
- **Size:** 500 training images and 100 validation images (123 phone boxes in validation).
- **Preparation:** labels were filtered to the `cell phone` class only, then exported in YOLO format (`dataset.yaml`, `images/`, `labels/`). No corrupt images were found.

## 4. Model and training

| Setting | Value |
|---|---|
| Base model | YOLO26n (Ultralytics), pretrained on COCO |
| Parameters | about 2.5 million |
| Classes | 1 (`cell phone`) |
| Epochs | 20 |
| Image size | 640 × 640 |
| Batch size | 8 |
| Optimizer | AdamW (chosen automatically, lr = 0.002) |
| Hardware | Google Colab, NVIDIA Tesla T4 |
| Training time | about 5.5 minutes (0.091 h) |

The first training run crashed the Colab session, most likely from running out of RAM. Lowering the batch size from 16 to 8 and using `workers=0` fixed it.

## 5. Results

Validation set (100 images, 123 phones), best checkpoint:

| Metric | Score |
|---|---|
| Precision | 0.550 |
| Recall | 0.366 |
| mAP@50 | 0.414 |
| mAP@50–95 | 0.238 |

**Interpretation:** about half of the predicted boxes are correct, and the model finds about a third of all phones in the validation set. Many COCO phones are small, far away or partly hidden, which makes this set hard. In the app's real use case, a phone held close to the webcam, detection worked well:

- Stock photo of a person holding a phone: detected, confidence 0.80
- Live webcam, phone held up: detected
- Live webcam, no phone: no false detection

## 6. Deployment

**Hugging Face model repo** stores two formats:
- `best.pt` (PyTorch), used by the Gradio Space
- `best.onnx` (ONNX, 9.4 MB), used by the browser app. Input `images [1, 3, 640, 640]`, output `output0 [1, 5, 8400]`.

**Vercel web app (`web/`)** runs the ONNX model in the browser with ONNX Runtime Web:
1. Each webcam frame is letterboxed to 640 × 640 and converted to an RGB tensor.
2. The model returns 8,400 candidate boxes (centre x, centre y, width, height, phone score).
3. Boxes below a 0.35 score are dropped, and non-maximum suppression (IoU 0.45) removes duplicates.
4. Boxes are mapped back to the video and drawn as "🦈 CHOMP!". The score increases once each time a phone appears.

**Privacy:** the webcam image never leaves the user's device. Only the model file is downloaded; no photos are uploaded or stored.

**Hugging Face Space (`hf-space/`)** is a simpler Gradio version that runs the PyTorch model on the server (ZeroGPU). Unlike the Vercel app, it sends the photo to Hugging Face's servers for processing.

## 7. Limitations

- Low recall (0.37): small, distant or sideways phones are often missed.
- Small dataset (500 images) and short training (20 epochs).
- COCO images are mostly everyday photos, not webcam selfies, so the training data doesn't fully match the app's use.
- Speed in the browser depends on the user's device, because the model runs on the CPU (WebAssembly).

## 8. Future work

- Train longer (50+ epochs) and on more images, including my own webcam photos.
- Add head-tracking (MediaPipe Face Landmarker) so the player steers a shark with their head and bites with their mouth, like the original fish game.
- Try WebGPU for faster in-browser inference.

## 9. Repository structure

```
shark-app/
├── README.md          ← this report
├── notebook/
│   └── train.ipynb    ← Colab: data, training, evaluation, export, upload
├── web/               ← Vercel app (root directory = web)
│   ├── index.html
│   └── app.js
└── hf-space/          ← Hugging Face Space (Gradio)
    ├── app.py
    ├── requirements.txt
    └── packages.txt
```

## 10. How to run

- **Web app:** open the Vercel link and allow camera access. To run it locally, serve the `web/` folder with any static server (for example `python -m http.server` inside `web/`) and open `http://localhost:8000`.
- **Retrain:** open `notebook/train.ipynb` in Google Colab with a T4 GPU and run all cells.

## 11. Credits

- Dataset: COCO 2017 (cocodataset.org). Annotations are CC BY 4.0; images come from Flickr under their own licences.
- Model and training: Ultralytics YOLO26.
- Dataset tooling: FiftyOne by Voxel51.
- Browser inference: ONNX Runtime Web.
- Idea: *You Are Fish* (you-are-fish.vercel.app).
