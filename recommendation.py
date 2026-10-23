# IMPORT LIBRARIES

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error

# STEP 1: LOAD DATASET

ratings = pd.read_csv(
    "data/u.data",
    sep="\t",
    names=["user_id", "movie_id", "ratings", "timestamp"]
)

# STEP 2: TRAIN / VALIDATION / TEST SPLIT

train_pool, test_data = train_test_split(
    ratings,
    test_size=0.2,
    random_state=42
)

train_data, val_data = train_test_split(
    train_pool,
    test_size=0.2,
    random_state=42
)

print("Training:", train_data.shape)
print("Validation:", val_data.shape)
print("Testing:", test_data.shape)

# STEP 3: CREATE RATINGS MATRIX

train_matrix = train_data.pivot_table(
    index="movie_id",
    columns="user_id",
    values="ratings"
)

train_matrix = train_matrix.reindex(
    index=sorted(ratings["movie_id"].unique()),
    columns=sorted(ratings["user_id"].unique())
)

Y = train_matrix.fillna(0).to_numpy(dtype=np.float32)
R = train_matrix.notna().to_numpy(dtype=np.float32)

Y_tf = tf.constant(Y, dtype=tf.float32)
R_tf = tf.constant(R, dtype=tf.float32)

print("Y shape:", Y.shape)
print("R shape:", R.shape)
print("Training ratings:", int(R.sum()))

# STEP 4: COLLABORATIVE FILTERING COST

def cofi_cost_func(X, W, b, Y, R, lambda_):

    predictions = tf.matmul(
        X, W, transpose_b=True
    ) + b

    errors = (predictions - Y) * R

    cost = 0.5 * tf.reduce_sum(
        tf.square(errors)
    )

    regularization = (lambda_ / 2) * (
        tf.reduce_sum(tf.square(W))
        + tf.reduce_sum(tf.square(X))
    )

    return cost + regularization

# STEP 5: HYPERPARAMETER TUNING

num_movies, num_users = Y.shape
num_features = 10

results = []

for lambda_ in [10.0, 30.0, 100.0]:

    print(f"\nTraining with lambda = {lambda_}")

    tf.random.set_seed(42)

    X = tf.Variable(
        tf.random.normal((num_movies, num_features))
    )

    W = tf.Variable(
        tf.random.normal((num_users, num_features))
    )

    b = tf.Variable(
        tf.zeros((1, num_users))
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.01
    )

    for iteration in range(200):

        with tf.GradientTape() as tape:

            cost = cofi_cost_func(
                X, W, b, Y_tf, R_tf, lambda_
            )

        gradients = tape.gradient(
            cost, [X, W, b]
        )

        optimizer.apply_gradients(
            zip(gradients, [X, W, b])
        )

    # Generate predictions
    predictions = (
        tf.matmul(X, W, transpose_b=True) + b
    ).numpy()

    predictions_df = pd.DataFrame(
        predictions,
        index=train_matrix.index,
        columns=train_matrix.columns
    )

    # Predict validation ratings
    val_predictions = [
        predictions_df.loc[movie_id, user_id]
        for user_id, movie_id in zip(
            val_data["user_id"],
            val_data["movie_id"]
        )
    ]

    val_predictions = np.clip(
        val_predictions, 1, 5
    )

    # Validation RMSE
    val_rmse = np.sqrt(
        mean_squared_error(
            val_data["ratings"],
            val_predictions
        )
    )

    results.append((lambda_, val_rmse))

    print(
        f"Lambda: {lambda_} | "
        f"Validation RMSE: {val_rmse:.4f}"
    )

# STEP 6: SELECT BEST LAMBDA

best_lambda, best_rmse = min(
    results,
    key=lambda x: x[1]
)

print("\nBest lambda:", best_lambda)
print("Best validation RMSE:", best_rmse)

# STEP 7: TRAIN FINAL MODEL

# Combine training and validation ratings
final_train_data = pd.concat(
    [train_data, val_data],
    ignore_index=True
)

print("\nFinal training ratings:", len(final_train_data))

# Rebuild the ratings matrix with 80,000 ratings
final_train_matrix = final_train_data.pivot_table(
    index="movie_id",
    columns="user_id",
    values="ratings"
)

final_train_matrix = final_train_matrix.reindex(
    index=sorted(ratings["movie_id"].unique()),
    columns=sorted(ratings["user_id"].unique())
)

Y_final = final_train_matrix.fillna(0).to_numpy(
    dtype=np.float32
)

R_final = final_train_matrix.notna().to_numpy(
    dtype=np.float32
)

Y_final_tf = tf.constant(Y_final, dtype=tf.float32)
R_final_tf = tf.constant(R_final, dtype=tf.float32)

# Initialize fresh parameters
tf.random.set_seed(42)

X_final = tf.Variable(
    tf.random.normal((num_movies, num_features))
)

W_final = tf.Variable(
    tf.random.normal((num_users, num_features))
)

b_final = tf.Variable(
    tf.zeros((1, num_users))
)

final_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.01
)

# Train with the best lambda
for iteration in range(200):

    with tf.GradientTape() as tape:
        final_cost = cofi_cost_func(
            X_final,
            W_final,
            b_final,
            Y_final_tf,
            R_final_tf,
            best_lambda
        )

    gradients = tape.gradient(
        final_cost,
        [X_final, W_final, b_final]
    )

    final_optimizer.apply_gradients(
        zip(gradients, [X_final, W_final, b_final])
    )

    if iteration % 20 == 0:
        print(
            f"Iteration {iteration}: "
            f"Cost = {final_cost.numpy():.2f}"
        )

print("\nFinal model training completed!")

# STEP 8: FINAL MODEL EVALUATION

# Generate predicted ratings
final_predictions = (
    tf.matmul(
        X_final,
        W_final,
        transpose_b=True
    ) + b_final
).numpy()

# Convert predictions to a DataFrame
final_predictions_df = pd.DataFrame(
    final_predictions,
    index=final_train_matrix.index,
    columns=final_train_matrix.columns
)

# Get predictions for the test dataset
test_predictions = [
    final_predictions_df.loc[movie_id, user_id]
    for user_id, movie_id in zip(
        test_data["user_id"],
        test_data["movie_id"]
    )
]

# Clip predictions to the valid rating range
test_predictions = np.clip(
    test_predictions, 1, 5
)

# Actual test ratings
actual_ratings = test_data["ratings"].to_numpy()

# Calculate final model metrics
test_mae = mean_absolute_error(
    actual_ratings,
    test_predictions
)

test_rmse = np.sqrt(
    mean_squared_error(
        actual_ratings,
        test_predictions
    )
)

# STEP 9: GLOBAL MEAN BASELINE

global_mean = final_train_data["ratings"].mean()

baseline_predictions = np.full(
    len(test_data),
    global_mean
)

baseline_mae = mean_absolute_error(
    actual_ratings,
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        actual_ratings,
        baseline_predictions
    )
)

# STEP 10: PRINT COMPARISON

print("\n--- FINAL MODEL RESULTS ---")

print(f"TensorFlow MAE: {test_mae:.4f}")
print(f"TensorFlow RMSE: {test_rmse:.4f}")

print(f"Baseline MAE: {baseline_mae:.4f}")
print(f"Baseline RMSE: {baseline_rmse:.4f}")

# STEP 11: REUSABLE RECOMMENDATION FUNCTION

# Load movie titles
movies = pd.read_csv(
    "data/u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=["movie_id", "title"]
)


def recommend_movies(user_id, top_n=10):

    # Check whether user exists
    if user_id not in final_predictions_df.columns:
        print(f"User {user_id} not found.")
        return None

    # Find movies already rated by the user
    rated_movies = ratings.loc[
        ratings["user_id"] == user_id,
        "movie_id"
    ]

    # Get predicted ratings for this user
    user_predictions = final_predictions_df[user_id]

    # Remove already rated movies
    unseen_predictions = user_predictions.drop(
        index=rated_movies
    )

    # Select highest predicted ratings
    top_movies = unseen_predictions.nlargest(top_n)

    # Clip predictions to valid range
    top_movies = top_movies.clip(lower=1, upper=5)

    # Build recommendations table
    recommendations = pd.DataFrame({
        "movie_id": top_movies.index,
        "predicted_rating": top_movies.values
    })

    # Add movie titles
    recommendations = recommendations.merge(
        movies,
        on="movie_id",
        how="left"
    )

    return recommendations[
        ["title", "predicted_rating"]
    ]

# STEP 12: TEST THE FUNCTION

print("\nRecommendations for User 1:")
print(recommend_movies(user_id=1, top_n=5))

print("\nRecommendations for User 10:")
print(recommend_movies(user_id=10, top_n=5))

print("\nRecommendations for User 50:")
print(recommend_movies(user_id=50, top_n=5))

# STEP 13: SAVE MODEL OUTPUTS

# Create models folder
os.makedirs("models", exist_ok=True)

# Save predicted ratings for every user and movie
final_predictions_df.to_pickle(
    "models/predicted_ratings.pkl"
)

print("Predicted ratings saved successfully!")
