import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

# preprocess the image in order to be infered by the model
def preprocessImage(img):
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
    img = transform(img).unsqueeze(0)
    return img

# predict the class of the image
def predict(img):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('aerialDelter.pth')
    model.eval()

    img = preprocessImage(img)

    with torch.no_grad():
        output = model(img)

    classes = ['grass', 'marshy', 'rocky', 'sandy']
    _, pred = output.max(1)
    return classes[pred.item()]

def main():

    page_bg_img = '''
    <style>
    body {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
    }
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    st.title('DELTER')
    st.write('DELTER stands for Deep Learning Model Terrain Recognition. A finetuned ResNet50 (using PyTorch Library) predicts the terrain of the given picture.')
    st.write('Upload an image and click on the predict button to see the result')

    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write('')
        label = predict(image)
        st.write(f'Prediction: {label}')
if __name__ == '__main__':
    main()