import streamlit as st 

# EDA Pkgs
import pandas as pd 
import numpy as np 
#Image

from PIL import Image
image=Image.open('Brainly_logo.png')
st.image(image,use_column_width=True)
# Utils
import joblib 

model = joblib.load(open("/Users/brl.314/Downloads/india_spam_june.pkl","rb"))

@st.cache(suppress_st_warning=True)

def pred(x):

	results = model.predict([x])
	
	return results[0]

#Confidence Function

def conf(x):
    
    confidence = model.decision_function([x])
    
    return confidence[0]

def main():
	with st.form(key='India Spam Classifier'):
		st.title("India Spam Classifier")
		raw_text = st.text_area("Enter Question","Enter Text Here")
		submit_text = st.form_submit_button(label='Submit')

	if submit_text:
		col1, col2, col3 = st.columns(3)

		prediction = pred(raw_text)
		confidence_score = round(abs(conf(raw_text)),3)

		with col1:
			st.success("Prediction")
			st.write(prediction)

		with col2:
			st.success("Confidence Score")
			st.write(confidence_score)

		with col3:
			st.success("Community Bot says")	
			if confidence_score > 0.7 and prediction=='Spam':
				st.write('Delete it 👎🏽')
			if confidence_score > 0.7 and prediction=='Not Spam':
				st.write('Keep it 👍🏽')
			if confidence_score < 0.7 and prediction=='Spam':
				st.write('Send it to a moderator 🥷🏼')
			if confidence_score < 0.7 and prediction=='Not Spam':
				st.write('Send it to a moderator 🥷🏼')

	
if __name__ == '__main__':
	main()