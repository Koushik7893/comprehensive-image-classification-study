# 📘 Multi-Perspective Image Classification: Binary, Multi-Class, Multi-Label & Transformers

---

## 📖 Overview / Introduction

This project implements various image classification tasks using both **TensorFlow** and **PyTorch**, covering:

* **Binary Classification**
* **Multi-Class Classification**
* **Multi-Label Classification**
* **Transformer-based Architectures**
* **Explainability with Grad-CAM and other visualization tools**

It serves as a comprehensive playground to explore different image classification scenarios with diverse datasets and model families — from CNNs to Transformers.

---

## 📂 Repository Structure

```bash
📦 image-classification-project/
├── binary_classification/
│   ├── binary_lenet_mnist_tensorflow.ipynb
│   ├── binary_lenet_mnist_pytorch.ipynb
│   ├── binary_pneumonia_cnn_tensorflow.ipynb
│   └── binary_pneumonia_cnn_pytorch.ipynb

├── multiclass_classification/
│   ├── multiclass_vgg_cifar10_tensorflow.ipynb
│   ├── multiclass_vgg_cifar10_pytorch.ipynb
│   ├── multiclass_resnet_cifar100_tensorflow.ipynb
│   ├── multiclass_resnet_cifar100_pytorch.ipynb
│   ├── multiclass_efficientnetb0_tiny_imagenet_tensorflow.ipynb
│   └── multiclass_efficientnetb0_tiny_imagenet_pytorch.ipynb

├── multilabel_classification/
│   ├── multilabel_custom_cnn_sigmoid_tensorflow.ipynb
│   └── multilabel_custom_cnn_bce_logit_pytorch.ipynb

├── explainability/
│   ├── brain_ct_gradcam_visualization_tensorflow.ipynb
│   └── chest_xray_gradcam_resnet_tensorflow.ipynb

├── transformer_models/
│   ├── vit_brain_cancer_mri_pytorch.ipynb
│   ├── swin_transformer_breast_cancer_pytorch.ipynb
│   └── inceptionresnetv2_brain_ct_tensorflow.ipynb

├── README.md
├── LICENSE
└── requirements.txt
```

---

## 🧩 Key Features

* Clean implementations of classic and modern CNNs (*LeNet, VGG, ResNet, EfficientNet*, etc.)
* Both **TensorFlow** and **PyTorch** versions for each task
* Support for **binary**, **multi-class**, and **multi-label** classification
* Transformer models like **ViT** and **Swin Transformer**
* **Grad-CAM** and advanced visualization tools for model interpretability
* Modular, organized Jupyter Notebook structure for easy learning and reference
* All notebooks are **Google Colab compatible**, with GPU support for training

Here’s the continuation of your README with:

* 📊 **Model Architecture & Training Details** (excluding explainability)
* ⚙️ **Installation / Setup**
* 🚀 **Usage**

---

## 🖼️ Sample Results / Visualizations

```markdown
![Grad-CAM Output](assets/sample_gradcam_output.png)
```

---

## 📊 Model Architecture & Training Details

| Notebook                                                   | Model                | Dataset               | Notes                      |
| ---------------------------------------------------------- | -------------------- | --------------------- | -------------------------- |
| `binary_lenet_mnist_tensorflow.ipynb`                      | LeNet-5              | MNIST (Even vs. Odd)  | Binary Classification      |
| `binary_lenet_mnist_pytorch.ipynb`                         | LeNet-5              | MNIST (Even vs. Odd)  | Binary Classification      |
| `binary_pneumonia_cnn_tensorflow.ipynb`                    | Custom CNN           | Chest X-ray           | Binary Classification      |
| `binary_pneumonia_cnn_pytorch.ipynb`                       | Custom CNN           | Chest X-ray           | Binary Classification      |
| `multiclass_vgg_cifar10_tensorflow.ipynb`                  | VGG                  | CIFAR-10              | Multi-class Classification |
| `multiclass_vgg_cifar10_pytorch.ipynb`                     | VGG                  | CIFAR-10              | Multi-class Classification |
| `multiclass_resnet_cifar100_tensorflow.ipynb`              | ResNet               | CIFAR-100             | Multi-class Classification |
| `multiclass_resnet_cifar100_pytorch.ipynb`                 | ResNet               | CIFAR-100             | Multi-class Classification |
| `multiclass_efficientnetb0_tiny_imagenet_tensorflow.ipynb` | EfficientNetB0       | Tiny ImageNet         | Multi-class Classification |
| `multiclass_efficientnetb0_tiny_imagenet_pytorch.ipynb`    | EfficientNetB0       | Tiny ImageNet         | Multi-class Classification |
| `multilabel_custom_cnn_sigmoid_tensorflow.ipynb`           | Custom CNN + Sigmoid | Custom Subset         | Multi-label Classification |
| `multilabel_custom_cnn_bce_logit_pytorch.ipynb`            | Custom CNN           | Custom Subset         | Multi-label Classification |
| `vit_brain_cancer_mri_pytorch.ipynb`                       | ViT (HuggingFace)    | Brain MRI             | Transformer-based Model    |
| `swin_transformer_breast_cancer_pytorch.ipynb`             | Swin Transformer     | Breast Cancer Dataset | Transformer-based Model    |
| `inceptionresnetv2_brain_ct_tensorflow.ipynb`              | InceptionResNetV2    | Brain CT              | Transformer-based Model    |


---

## ⚙️ Installation / Setup

```bash
# Clone the repo
git clone https://github.com/your-Koushik7893/comprehensive-image-classification-study.git
cd comprehensive-image-classification-study

# Install dependencies
pip install -r requirements.txt
```

✅ **Google Colab Compatible**
To run on Colab:

1. Upload the desired notebook.
2. Mount Google Drive if using large datasets.
3. Install missing dependencies via `pip` in Colab.

---

## 🚀 Usage

* Each notebook is **standalone**.
* Open a notebook like `binary_lenet_mnist_tensorflow.ipynb`.
* Run the cells sequentially for:

  * Training
  * Evaluation
  * Visualization (Grad-CAM or metrics if applicable)

For visualization-based interpretation, refer to notebooks inside the `explainability/` folder.

---

## 🔍 Explainability / Interpretability

We used various interpretability tools to visualize what the models learn:

* **Grad-CAM**
* **Grad-CAM++**
* **Score-CAM**
* **SmoothGrad-CAM++**

These techniques helped us:

* Highlight **activated regions** in the image.
* Understand **class-specific attention maps**.
* Analyze **model bias**, overfitting signs, and **decision boundaries** visually.

The visualizations are especially useful in medical imaging tasks (e.g., Brain CT, Chest X-ray) to verify whether models focus on clinically meaningful regions.

---

## 🧰 Tools & Libraries

This project uses the following key tools and libraries:

* **Deep Learning Frameworks**:
  `TensorFlow`, `tf.keras`, `PyTorch`, `torchvision`, `timm`

* **Computer Vision Utilities**:
  `OpenCV`, `NumPy`, `Matplotlib`, `Seaborn`

* **Explainability Packages**:
  `tf-keras-vis`, `pytorch-gradcam`, `custom Grad-CAM scripts`

* **Development Environment**:
  Fully supported in **Google Colab** (with GPU runtime)

---

## 📌 Future Work / To-Do

* [ ] Add **Swin Transformer** implementation in TensorFlow
* [ ] Integrate **Noisy Student Training** for semi-supervised learning
* [ ] Include **dataset download automation scripts** for large datasets

---

## 🧑‍💻 Author / Contributors

**Koushik Reddy**
🔗 [Hugging Face](https://huggingface.co/Koushim) | [LinkedIn](https://www.linkedin.com/in/koushik-reddy-k-790938257)

Feel free to reach out for collaboration, suggestions, or questions!

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 💡 Acknowledgements / References

* [PyTorch](https://pytorch.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [Kaggle Datasets](https://www.kaggle.com/datasets)
* [Grad-CAM: Visual Explanations](https://arxiv.org/abs/1610.02391)
* [Hugging Face Transformers](https://huggingface.co/)
* [timm: PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)


