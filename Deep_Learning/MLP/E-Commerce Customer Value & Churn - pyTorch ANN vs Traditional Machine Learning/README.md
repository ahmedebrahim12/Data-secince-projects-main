# 🛒 E-Commerce Customer Value Prediction & Model Comparison

This project predicts **customer lifetime value (target_customer_value)** for an e-commerce dataset with **~250 features**.  
It compares **Traditional Machine Learning** models with a **Deep Learning (PyTorch ANN)** to explore how well neural networks perform on large tabular data.


## 📂 Project Structure


|   File  | Description |
|--------|-------------|
| **e_commerce_churn_(pytorch).py** | Main code: data prep, models, training & evaluation |
| **results/** |Saved figures and ONNX model |
| **README.md** | This file |





## 🚀 Key Steps

### 1️⃣ Data Preparation
* Loaded and cleaned a wide e-commerce dataset (~250 numeric & categorical features).
* Scaled continuous variables and encoded categorical features.
* Split into **train/test** sets.

### 2️⃣ Models Trained
| Model | Framework | Goal |
|-------|----------|------|
| **Linear Regression** | scikit-learn | Baseline regression |
| **Random Forest Regressor** | scikit-learn | Strong traditional ML benchmark |
| **Artificial Neural Network (ANN)** | PyTorch | Deep learning approach for tabular data |



## ⚙️ PyTorch ANN
* **Architecture**: 4 fully connected layers (250 → 128 → 64 → 32 → 1)  
* **Activations**: ReLU  
* **Regularization**: Dropout & Batch Normalization  
* **Optimizer**: Adam (`lr=0.001`)  
* **Loss**: MSE  
* **Early Stopping**: Stops training when validation loss stops improving.

![Visual Comparison](https://github.com/ahmedebrahim12/Data-secince-projects-main/blob/main/Deep_Learning/MLP/E-Commerce%20Customer%20Value%20&%20Churn%20-%20pyTorch%20ANN%20vs%20Traditional%20Machine%20Learning/Training%20and%20Test%20Loss,%20R2,%20and%20MAE%20over%20Epochs.png?raw=true)


## 📊 Results

| Model               | MSE        | MAE     | R²      | Training Time |
|---------------------|-----------:|-------:|-------:|-------------:|
| Linear Regression   | 196,136    | 166.36 | 0.697  | **~2 sec**   |
| Random Forest       | 146,874    | 124.28 | 0.773  | ~5,000 sec   |
| **PyTorch ANN**     | **123,118**| 130.23 | **0.810** | ~2,940 sec |

### Visual Comparison  
Four bar charts show side-by-side metrics (**MSE, MAE, R², Training Time**) for all models.

![Visual Comparison](https://github.com/ahmedebrahim12/Data-secince-projects-main/blob/main/Deep_Learning/MLP/E-Commerce%20Customer%20Value%20&%20Churn%20-%20pyTorch%20ANN%20vs%20Traditional%20Machine%20Learning/Comparing%20models%20across%20the%20four%20metrics.png?raw=true)


## 🧠 Insights
* **Traditional ML often leads** on structured/tabular data.  
* **ANN outperformed Random Forest here**, likely due to:
  - High feature count (~250) capturing complex nonlinear relationships.
  - Careful tuning & regularization.
* Still, Random Forest required far **less tuning** and trained faster.


## 📦 Deployment
The trained PyTorch model is exported to **ONNX** for cross-platform inference:
```python
torch.onnx.export(model, example_input, "ecommerce_model.onnx", input_names=['input'], output_names=['output'])
```



## 🏆 Conclusion

While classical ML is typically dominant for tabular data,
this experiment shows that deep learning can match or exceed it when features are numerous and interactions complex.
Neural networks shine even more on unstructured data (images, audio, text)—the focus of upcoming projects.

## 🔧 Tech Stack

- Python 3.x

- pandas, numpy, scikit-learn

- PyTorch

- plotly / matplotlib

- onnxruntime

## Running Tests

To run tests, run the following command

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run training & evaluation
python e_commerce_churn_(pytorch).py

```




## How to use
1. Save the text above as **`README.md`** in your project root.
2. Adjust dataset path / repo links if needed.
3. Commit & push to GitHub – GitHub will render it beautifully.

This README gives visitors:
* **Clear context** (what & why).  
* **Quick start instructions**.  
* **Metrics & visual insight**.  
* **Conclusion** highlighting key learnings.




