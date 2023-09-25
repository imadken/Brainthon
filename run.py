import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras


products_encoded = pd.read_csv("products_encoded.csv")
I = pd.read_csv("products_new.csv")
# Load the model from the file
model = tf.keras.models.load_model('my_model.h5')

# Use the loaded model for inference
def model_preds(p1,p2,p3):
  data = {
    'Column1': [p1],
    'Column2': [p2],
    'Column3': [p3]
  }
  data= pd.DataFrame(data)
  # Define different combinations of features and targets
  X1 = data.iloc[:, [0, 1]].to_numpy()

  X2 = data.iloc[:, [0, 2]].to_numpy()

  X3 = data.iloc[:, [1, 2]].to_numpy()

  X4 = data.iloc[:, [1, 0]].to_numpy()

  X5 = data.iloc[:, [2, 0]].to_numpy()

  X6 = data.iloc[:, [2, 1]].to_numpy()

  # Concatenate the feature sets vertically using NumPy
  X = np.vstack([X1, X2, X3, X4, X5, X6])
  # Convert X and y back to DataFrames
  X_df = pd.DataFrame(X, columns=["Feature1", "Feature2"])
  # Merge dataset1 with products based on 'product_1_id'
  X_df_marged = X_df.merge(products_encoded, left_on='Feature1', right_on='id', how='left',
                    suffixes=('', '_feature1'))


  # Drop the 'id' column as it's no longer needed
  X_df_marged = X_df_marged.drop(columns=['id'])

  # Repeat the same process for 'product_2_id'
  X_df_marged = X_df_marged.merge(products_encoded,
                    left_on='Feature2', right_on='id', how='left',
                    suffixes=('', '_feature2'))



  # Drop the 'id' column as it's no longer needed
  X_df_marged = X_df_marged.drop(columns=['id','Feature1',	'Feature2'])

  # Make predictions using the model
  predictions = model.predict(X_df_marged)


  # Find the indices of the top 11 classes with the highest probability for each prediction
  top11_indices = np.argsort(predictions, axis=1)[:, -11:][:, ::-1]
  predicted_indices = top11_indices.transpose().flatten()

  predicted_indices = np.unique(predicted_indices)
  predicted_indices = predicted_indices[predicted_indices != p1]
  predicted_indices = predicted_indices[predicted_indices != p2]
  predicted_indices = predicted_indices[predicted_indices != p3]

  # Map the predicted indices to product names using the product mapping DataFrame
  predicted_product_names = I.loc[predicted_indices,'real_name'].values
  predicted_product_image = I.loc[predicted_indices,'image'].values

  # Create a new DataFrame with the predicted product names
  predicted_df = pd.DataFrame({
      'Predicted_Product_Id': predicted_indices,
      'Predicted_Product_Name': predicted_product_names,
      'Predicted_Product_Image': predicted_product_image})

  return predicted_df[:8]
