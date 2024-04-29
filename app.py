import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import io
import time
import matplotlib.pyplot as plt

st.set_option('deprecation.showfileUploaderEncoding', False)

def import_and_predict(image_data, model):
    start_time = time.time()
    image = ImageOps.fit(image_data, (100,100))
    image = image.convert('RGB')
    image = np.asarray(image)
    st.image(image, channels='RGB')
    image = (image.astype(np.float32) / 255.0)
    img_reshape = image[np.newaxis,...]
    prediction = model.predict(img_reshape)
    end_time = time.time()
    process_time = end_time - start_time
    return prediction, process_time

model = tf.keras.models.load_model('new_model.h5')

st.write("""
         # **Glaucoma detector**
         """
         )

st.write("This is a simple image classification web app to predict glaucoma through a fundus image of the eye")

file = st.file_uploader("Please upload a JPG image file", type=["jpg"])

if file is None:
    st.text("You haven't uploaded a jpg image file")
else:
    imageI = Image.open(file)
    prediction, process_time = import_and_predict(imageI, model)
    pred = prediction[0][0]
    st.write(f"**Probability of being Healthy:** {pred:.2%}")
    st.write(f"**Probability of being affected by Glaucoma:** {(1-pred):.2%}")

    st.bar_chart({"Healthy": pred, "Glaucoma": 1-pred})
    if pred > 0.5:
        st.write("""
                 ## *Prediction:* Your eye is Healthy. Great!!
                 """
                 )
        st.balloons()
    else:
        st.write("""
                 ## *Prediction:* You are affected by Glaucoma. Please consult an ophthalmologist as soon as possible.
                 """
                 )

    st.write(f"Processing time: {process_time:.2f} seconds")

    # Pie chart
    labels = ['Healthy', 'Glaucoma']
    sizes = [pred, 1-pred]
    colors = ['lightgreen', 'lightcoral']
    explode = (0.1, 0)
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    st.pyplot(plt)

# Download button
if st.button('Press to download predictions'):
    result_text = f"Probability of being Healthy: {pred:.2%}\nProbability of being affected by Glaucoma: {(1-pred):.2%}\nProcessing time: {process_time:.2f} seconds"
    result_bytes = result_text.encode()
    result_io = io.BytesIO(result_bytes)
    st.download_button(label="Download Result", data=result_io, file_name='prediction_result.txt', mime='text/plain')
