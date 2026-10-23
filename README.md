
# 🎬 CineMatch — Personalized Movie Recommendation System

An end-to-end machine learning project that recommends movies based on user preferences using **TensorFlow Collaborative Filtering and Matrix Factorization**.

CineMatch features a Netflix-inspired Streamlit interface where users can explore personalized movie recommendations, view predicted ratings, and discover movies they have not watched.

## 🌐 Live Demo

🎬 **[Launch CineMatch — Live App](https://movie-recommendation-system-cinematch-ml.streamlit.app/)**

💻 **[View GitHub Repository](https://github.com/udhayakumar268/movie-recommendation-system)**

Explore personalized movie recommendations for existing MovieLens users through an interactive Streamlit application.

## 📌 Project Overview

CineMatch predicts how much a user might enjoy movies they have not rated by learning patterns from historical user-movie interactions.

The recommendation engine uses collaborative filtering with matrix factorization to learn latent features representing movies and user preferences.

### Machine Learning Workflow

1. Data loading and preprocessing
2. Train, validation, and test splitting
3. User-movie rating matrix creation
4. Collaborative filtering model development
5. Hyperparameter tuning
6. Final model training
7. Model evaluation and baseline comparison
8. Personalized recommendation generation
9. Saving predictions for deployment
10. Interactive Streamlit application

## 📊 Dataset

**MovieLens 100K Dataset**

| Property | Value |
|---|---|
| Total Ratings | 100,000 |
| Total Users | 943 |
| Total Movies | 1,682 |
| Rating Scale | 1–5 |
| Dataset Source | GroupLens Research |

**Dataset:** https://grouplens.org/datasets/movielens/100k/

### Dataset Splitting

| Dataset | Ratings | Percentage |
|---|---:|---:|
| Training | 64,000 | 64% |
| Validation | 16,000 | 16% |
| Testing | 20,000 | 20% |

After selecting the best hyperparameter, the training and validation datasets were combined to train the final model using **80,000 ratings**.

The remaining 20,000 ratings were reserved for final testing.

## 🛠️ Technologies Used

**Programming**
- Python

**Machine Learning**
- TensorFlow
- Collaborative Filtering
- Matrix Factorization
- L2 Regularization
- Adam Optimizer

**Data Processing and Evaluation**
- NumPy
- Pandas
- Scikit-learn

**Application Development**
- Streamlit
- HTML
- CSS

**Version Control and Deployment**
- Git
- GitHub
- Streamlit Community Cloud

## 🧠 Machine Learning Approach

### Collaborative Filtering

Collaborative filtering predicts user preferences by learning patterns from historical ratings across multiple users and movies.

CineMatch uses matrix factorization to represent movies and users through latent features.

### Prediction Formula

The predicted rating is calculated using:

$$
\hat{r}_{ij} = x_i^T w_j + b_j
$$

Where:

- \(x_i\) represents the latent features of movie \(i\).
- \(w_j\) represents the latent preferences of user \(j\).
- \(b_j\) represents the bias of user \(j\).

### Cost Function

The model minimizes squared prediction error over observed ratings while applying L2 regularization to movie and user latent factors.

A binary rating mask ensures that missing ratings are excluded from the training error.

### Training Configuration

| Parameter | Value |
|---|---|
| Latent Features | 10 |
| Optimizer | Adam |
| Learning Rate | 0.01 |
| Training Iterations | 200 |
| Regularization Values Tested | 10, 30, 100 |
| Best Regularization (λ) | 10 |
| Random Seed | 42 |

The regularization parameter was selected using validation RMSE.

The final model was retrained using the combined training and validation datasets.

## 📈 Model Performance

The final TensorFlow collaborative filtering model was evaluated on the held-out test dataset.

| Metric | TensorFlow Model | Global Mean Baseline |
|---|---:|---:|
| Mean Absolute Error (MAE) | **0.8788** | 0.9420 |
| Root Mean Squared Error (RMSE) | **1.1151** | 1.1239 |

### Key Findings

- Achieved a test MAE of **0.8788**.
- Achieved a test RMSE of **1.1151**.
- Selected **λ = 10** based on validation RMSE.
- Outperformed the global mean baseline on both evaluation metrics.
- Used 80,000 ratings to train the final model.

Predicted ratings are clipped to the valid 1–5 rating scale during evaluation and when displayed in the application.

## 🎯 Personalized Recommendation System

The recommendation engine follows these steps:

1. Select an existing MovieLens user.
2. Retrieve predicted ratings for all movies.
3. Identify movies already rated by the selected user.
4. Remove previously rated movies from the recommendation candidates.
5. Rank unseen movies by their predicted scores.
6. Return the highest-scoring movies with their titles and predicted ratings.

The application allows users to switch between profiles and customize the number of recommendations displayed.

## 🎨 Streamlit Application

CineMatch provides a Netflix-inspired interface with a dark cinematic theme.

### Application Features

- Personalized recommendations for existing MovieLens users
- User profile selection
- Featured movie banner
- Horizontally scrollable movie cards
- Clickable movie details popup
- Predicted movie ratings
- Movie release years
- Previously highly rated movies
- Model performance metrics
- Adjustable recommendation count

The application loads the saved prediction matrix directly, allowing recommendations without retraining the TensorFlow model.

**Live Application:** https://movie-recommendation-system-cinematch-ml.streamlit.app/

## 📁 Project Structure

```text
movie-recommendation-system/
│
├── app.py
├── recommendation.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── u.data
│   └── u.item
│
└── models/
    └── predicted_ratings.pkl
```

### File Descriptions

| File | Purpose |
|---|---|
| `app.py` | Streamlit movie recommendation interface |
| `recommendation.py` | TensorFlow collaborative filtering training pipeline |
| `requirements.txt` | Python dependencies |
| `data/u.data` | MovieLens user ratings |
| `data/u.item` | Movie titles and metadata |
| `models/predicted_ratings.pkl` | Saved user-movie prediction matrix |

## 🚀 Running the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/udhayakumar268/movie-recommendation-system.git
cd movie-recommendation-system
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

Streamlit will display a local URL where you can access CineMatch.

### 5. Retrain the Model (Optional)

To reproduce the training workflow:

```bash
python recommendation.py
```

The training script will:

- Load MovieLens ratings
- Tune the regularization parameter
- Train the final TensorFlow model
- Evaluate model performance
- Generate personalized recommendations
- Save predictions to `models/predicted_ratings.pkl`

Running this command will overwrite the existing saved prediction matrix.

## ⚠️ Model Limitations

- The model supports existing MovieLens users rather than entirely new user profiles.
- Recommendations depend on historical rating patterns.
- The model does not currently use movie descriptions, genres, or other content features.
- Predicted ratings may exceed the 1–5 range before clipping.
- The current application uses styled movie cards instead of real movie posters.
- Ranking-specific metrics such as Precision@K and Recall@K have not yet been evaluated.

## 🔮 Future Improvements

- Develop a content-based recommendation model
- Build a hybrid recommendation system
- Integrate real movie posters and additional metadata
- Implement cold-start recommendations
- Evaluate ranking performance using Precision@K and Recall@K
- Explore neural collaborative filtering
- Improve hyperparameter optimization

## 📚 Dataset Attribution

This project uses the MovieLens 100K dataset provided by GroupLens Research.

Dataset website:
https://grouplens.org/datasets/movielens/100k/

Users should review the dataset's license and usage conditions before redistributing it.

## 👨‍💻 Author

**Udhaya Kumar**

MSc Artificial Intelligence — University of South Wales

Machine learning enthusiast with hands-on experience building end-to-end machine learning applications using Python, TensorFlow, scikit-learn, and Streamlit.

**Project:** CineMatch — Personalized Movie Recommendation System
