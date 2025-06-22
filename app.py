import streamlit as st
import torch
import matplotlib.pyplot as plt
from generator import load_generator, one_hot
import numpy as np

# Load model
generator = load_generator()
noise_dim = 100

st.title("🖌️ Handwritten Digit Generator")
st.write("Select a digit and generate 5 handwritten samples using a trained model.")

digit = st.selectbox("Choose a digit (0-9):", list(range(10)))

if st.button("Generate Images"):
    with st.spinner("Generating..."):
        z = torch.randn(5, noise_dim)
        labels = torch.tensor([digit]*5)
        onehot = one_hot(labels)

        with torch.no_grad():
            fake_imgs = generator(z, onehot).view(-1, 28, 28)

        # Display images
        st.write(f"Generated samples for digit: {digit}")
        cols = st.columns(5)
        for i in range(5):
            img = fake_imgs[i].numpy()
            cols[i].image(img, width=100, clamp=True)
