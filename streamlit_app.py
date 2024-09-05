import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App')

st.info('This is app builds a machine learning model')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
  df

  st.write('**X**')
  X = df.drop('species', axis=1)
  X

  st.write('**Y**')
  y = df.species
  y
  
with st.expander('Data visualization'):
  # bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","sex"
  st.scatter_chart(data=df,x='bill_length_mm',y='body_mass_g', color='species')

# Data Preparation
with st.sidebar:
  st.header('Input Features')
  island = st.selectbox('Island', ('Torgersen','Dream','Biscoe'))
  gender = st.selectbox('Gender',('male','female'))
  bill_length_mm = st.slider('Bill length (mm)',32.1,59.6,43.9)
  flipper_length_mm = st.slider('Flipper length (mm)',172.0, 231.0, 201.0)
  body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)

  #Create a Dataframe for the input features
  data = {'island':island,
          'bill_length_mm':bill_length_mm,
          'flipper_length_mm':flipper_length_mm,
          'body_mass_g':body_mass_g,
          'gender':gender}
  input_df = pd.DataFrame(data,index=[0])
  input_penguins = pd.concat([input_df,X],axis=0)

with st.expander('Input features'):
  st.write('**Input penguins***') 
  input_df
  st.write('**(combined penguins data)**')
  input_penguins
#input_df

#Encode
encode = ['island','sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)
