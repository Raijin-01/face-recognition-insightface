# Pretrained Face Recognition with InsightFace

Real-time face recognition in Google Colab using a pretrained InsightFace `buffalo_l` model and ArcFace embeddings.

## Pipeline

Webcam -> Face Detection -> ArcFace Embedding -> Cosine Similarity -> Identity / Unknown

## Structure

```text
face-recognition-insightface/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── face_recognition_colab.ipynb
├── src/
│   ├── __init__.py
│   ├── recognizer.py
│   └── colab_webcam.py
├── data/
│   └── known_faces/
│       └── .gitkeep
└── outputs/
    └── .gitkeep
```

## Setup in Colab

```python
!git clone https://github.com/Raijin-01/face-recognition-insightface.git
%cd face-recognition-insightface
!pip install -r requirements.txt
```

The pretrained InsightFace model is downloaded automatically on first use.

## Reference faces

Place reference images under:

```text
data/known_faces/<person_id>/
```

Example:

```text
data/known_faces/
├── person_01/
│   ├── image_01.jpg
│   ├── image_02.jpg
│   └── image_03.jpg
└── person_02/
    ├── image_01.jpg
    └── image_02.jpg
```

Use IDs/names you are comfortable publishing. Real face images and generated embeddings are excluded from Git by `.gitignore`.

## Recognition

Reference images are converted into ArcFace embeddings and averaged into one normalized reference embedding per identity. Live webcam faces are embedded and compared using cosine similarity.

The recognition threshold should be calibrated on validation data rather than treated as a universal constant.

## Privacy

Face images and biometric embeddings should be treated as private data. Do not commit real reference images, embeddings, or other biometric records to a public repository.
