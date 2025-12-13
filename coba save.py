import tensorflow as tf
MODEL_PATH = 'chikinspect_model_cropped_final.keras'
model = tf.keras.models.load_model(MODEL_PATH)
model.save("chikinspect_model_cropped_final.keras", save_format="keras")
