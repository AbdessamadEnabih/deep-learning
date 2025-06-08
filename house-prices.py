import numpy as np
import tensorflow as tf
# import test_house_prices



"""
Formula for calculating the price of a house based on its features.

base house = 50k
every bedroom = 50k

if house with 2 bedrooms, price = 50k + 2 * 50k = 150k
"""

def create_training_data():
    n_bedrooms = None
    price_in_hunder_thousands = None
    
    n_bedrooms = np.arange(1, 7,dtype=np.float32)
    price_in_hunder_thousands = (np.arange(1, 7, dtype=np.float32) * 50 + 50) / 100

    return n_bedrooms, price_in_hunder_thousands


features, targets = create_training_data()

print("Features (n_bedrooms):", features)
print("Targets (price_in_hundreds_of_thousands):", targets)

def define_and_compile_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,), name='input_layer'),
        tf.keras.layers.Dense(1, name='output_layer', activation='linear')
    ])
    
    model.compile(optimizer='sgd',loss='mean_squared_error')
        
    return model


untrained_model = define_and_compile_model()

untrained_model.summary()

def train_model():
    
    n_bedrooms, price_in_hundreds_of_thousands = create_training_data()
    model = define_and_compile_model()
    
    model.fit(n_bedrooms, price_in_hundreds_of_thousands, epochs=500, verbose=1)
    
    return model


trained_model = train_model()

new_n_bedrooms = np.array([7.0])
predicted_price = trained_model.predict(new_n_bedrooms).item()
print(f"Your model predicted a price of {predicted_price:.2f} hundreds of thousands of dollars for a {int(new_n_bedrooms.item())} bedrooms house")
