# Multimodal Movie Genre Classification Report

**Group:** LTH (252)  
**Course:** Deep Learning (CO3021)  
**Institution:** Ho Chi Minh City University of Technology (HCMUT)

---

## 1. Dataset Exploration & Preprocessing

### 📊 Dataset Overview: MMIMDb

The project utilizes the **MMIMDb (Multi-Modal IMDb)** dataset, a standard benchmark for multimodal classification tasks. The primary objective is to predict movie genres by integrating visual information (movie posters) and textual information (plot summaries).

- **Data Source:** [Hugging Face - MMIMDb Dataset](https://huggingface.co/datasets/sxj1215/mmimdb)
- **Problem Type:** Multi-label classification

#### Key Statistics:

| Metric                      | Value                              |
| :-------------------------- | :--------------------------------- |
| **Total Samples**           | 15,552                             |
| **Unique Classes (Genres)** | 26                                 |
| **Modalities**              | Visual (Posters) & Textual (Plots) |

#### Data Characteristics:

- **Textual Data:** The `plot` attribute provides a concise summary of the movie's main storyline.
- **Visual Data:** The `images` attribute contains official movie posters, reflecting the artistic style and color palette characteristic of each genre.
- **Label Distribution:** The most frequent genres are **Drama** (8,424 samples), **Comedy** (5,108 samples), and **Romance** (3,226 samples).

---

### 🖼️ Dataset Preview

The MMIMDb dataset structure combines textual content with visual imagery:

![Dataset Sample Preview](./public/multimodal/dataset.png)  
_Figure 1: Preview of the MMIMDb dataset structure displaying plot summaries and corresponding movie posters._

---

[⬅️ Back to README](./README.md)
