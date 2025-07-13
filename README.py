# readme_content = """
# 🎬 Movie Recommendation System

# Welcome to the **Movie Recommendation System**!  
# This Streamlit-based web app suggests movies using two intelligent approaches:

# - 🧠 **Content-Based Filtering**
# - 🤖 **Collaborative Filtering**

# ---

## 🚀 Demo

# <img src="https://img.shields.io/badge/Made%20with-Streamlit-orange?logo=streamlit" height="25"/>
# <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python" height="25"/>

# > **✨ Get personalized movie suggestions based on your preferences or what others love!**

# ---

## 📂 Project Structure


# 📁 model/
# ├── movie_list.pkl
# ├── similarity.pkl
# ├── movierecommendation_list.pkl
# ├── movies.pkl
# ├── final_dataset.pkl
# ├── knn_model.pkl
# └── csr_data.pkl

# 📄 app.py # Main Streamlit app
# 📄 index.py # Script to train Collaborative Filtering model
# 📄 movierecommendersystem.ipynb # Script to train Content-Based Filtering model
# 📄 README.md

# yaml
# Copy code


## ⚙️ Features

# | Feature                         | Description |
# |-------------------------------|-------------|
# | 🎬 **Content-Based Filtering** | Recommends movies similar to the one you select. |
# | 🧑‍🤝‍🧑 **Collaborative Filtering** | Suggests movies based on user ratings using KNN. |
# | 🎨 **Dark Themed UI**          | Stylish and intuitive user interface using Streamlit and CSS. |


## 🧰 How to Run the Project

### 1. 📦 Install Dependencies

# ```bash
# pip install -r requirements.txt



# 2. 🛠️ Generate Model Files (.pkl)

# Train collaborative filtering model
# python index.py

# Train content-based model
# Either run the notebook or convert it to script:
# jupyter notebook movierecommendersystem.ipynb


# 3. ▶️ Run the Streamlit App

# streamlit run app.py
# or 
# python -m streamlit run app.py

# 📊 Algorithms Used
# Content-Based Filtering: Cosine similarity on TF-IDF or metadata features.

# Collaborative Filtering: K-Nearest Neighbors using a user-item rating matrix.



