!pip install git+https://github.com/python-visualization/folium

import numpy as np
import math
import pandas as pd
import folium
import json
import os
import io

utr1 = pd.read_csv('utr1.csv',header=None)
utr =np.zeros((30,1))
utr[:] = utr1[:]

norm1 = pd.read_csv('norm1.csv',header=None)
norm =np.zeros((30,1))
norm[:] = norm1[:]

def sigmoid(z):

    s = 1 / (1 + np.exp(-z))

    return s

def sigmoiderivative(Z):

  f = Z * (1-Z)

  return f

def tanh(z):

    s = (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))

    return s

def tanhderivative(x):

    p = 1 - np.square(x)

    return p

def relu(z):

    s = np.maximum(0,z)

    return s

def reluderivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    return np.exp(x) / np.sum(e_x,axis=0)

layer = [30,20,15,6,5]
keep_prob = [1.0,1.0,1.0,1.0,1.0]

l = len(layer)
parameters = {}
np.random.seed(0)
for i in range(1,l-1):
    parameters["W"+str(i)]=np.zeros((layer[i],layer[i-1]))
    parameters["b"+str(i)]=np.zeros((layer[i],1))

parameters["W"+str(l-1)]=np.zeros((layer[l-1],layer[l-2]))
parameters["b"+str(l-1)]=np.zeros((layer[l-1],1))

w1 = pd.read_csv('W1.csv',header=None)
w12=np.zeros((20,30))
w12[:]=w1[:]
parameters['W1'][:]= w12[:]

w2 = pd.read_csv('W2.csv',header=None)
w22=np.zeros((15,20))
w22[:]=w2[:]
parameters['W2'][:]= w22[:]

w3 = pd.read_csv('W3.csv',header=None)
w32=np.zeros((6,15))
w32[:]=w3[:]
parameters['W3'][:]= w32[:]

w4 = pd.read_csv('W4.csv',header=None)
w42=np.zeros((5,6))
w42[:]=w4[:]
parameters['W4'][:]= w42[:]

b1 = pd.read_csv('b1.csv',header=None)
b12=np.zeros((20,1))
b12[:]=b1[:]
parameters['b1'][:]= b12[:]

b2 = pd.read_csv('b2.csv',header=None)
b22=np.zeros((15,1))
b22[:]=b2[:]
parameters['b2'][:]= b22[:]

b3 = pd.read_csv('b3.csv',header=None)
b32=np.zeros((6,1))
b32[:]=b3[:]
parameters['b3'][:]= b32[:]

b4 = pd.read_csv('b4.csv',header=None)
b42=np.zeros((5,1))
b42[:]=b4[:]
parameters['b4'][:]= b42[:]

def fwd_propagation(X,parameters,layer):

  l = len(layer)
  forward = {}

  forward["Z"+str(1)]=np.dot(parameters["W"+str(1)],X)+parameters["b"+str(1)]

  forward["A"+str(1)]=relu(forward["Z"+str(1)])

  for i in range(2,l-1):
    forward["Z"+str(i)]=np.dot(parameters["W"+str(i)],forward["A"+str(i-1)])+parameters["b"+str(i)]
    forward["A"+str(i)]=relu(forward["Z"+str(i)])

  forward["Z"+str(l-1)]=np.dot(parameters["W"+str(l-1)],forward["A"+str(l-2)])+parameters["b"+str(l-1)]
  forward["A"+str(l-1)]=sigmoid(forward["Z"+str(l-1)])

  return forward

  def predict(X,parameters,layer,keep_prob):
    l= len(layer)
    #fwd = fwd_propagation_drop(X,parameters,layer,keep_prob)
    fwd = fwd_propagation(X,parameters,layer)
    Y1 = fwd["A"+str(l-1)]
    Y2=np.argmax(Y1,axis=0)

    return Y2

sts = input("Name of the state? ")
yar = input("Enter the Year for prediction ")
print("\n")
print("The Entered state is = "+sts)
print("\n")
print("And The Entered Year is = "+yar)

if sts == "Arunanchal Pradesh":
  cap = 1
elif sts == "Andhra Pradesh":
  cap = 0
elif sts == "Assam":
  cap = 2
elif sts == "Bihar":
  cap = 3
elif sts == "Chhattisgarh":
  cap = 4
elif sts == "Delhi":
  cap = 5
elif sts == "Goa":
  cap = 6
elif sts == "Gujarat":
  cap = 7
elif sts == "Haryana":
  cap = 8
elif sts == "Himachal Pradesh":
  cap = 9
elif sts == "Jammu and Kashmir":
  cap = 10
elif sts == "Jharkhand":
  cap = 11
elif sts == "Karnataka":
  cap = 12
elif sts == "Kerala":
  cap = 13
elif sts == "Madhya Pradesh":
  cap = 14
elif sts == "Maharashtra":
  cap = 15
elif sts == "Manipur":
  cap = 16
elif sts == "Meghalaya":
  cap = 17
elif sts == "Mizoram":
  cap = 18
elif sts == "Nagaland":
   cap = 19
elif sts == "Orissa":
  cap = 20
elif sts == "Punjab":
  cap = 21
elif sts == "Rajasthan":
  cap = 22
elif sts == "Sikkim":
  cap = 23
elif sts == "Tamil Nadu":
  cap = 24
elif sts == "Tripura":
  cap = 25
elif sts == "Uttar Pradesh":
  cap = 26
elif sts == "Uttaranchal":
  cap = 27
elif sts == "West Bengal":
  cap = 28
elif sts == "All":
  cap = 29
else:
  print("Incorrect Names Entered \n")
  exit()

Tx = pd.read_csv('labelled_input1.csv')

tstx = np.zeros((29,30))
tstx[:] = Tx[:]

ad1 = np.zeros((1,30))
q = yar
ad1[[0],[0]] = q

tstx = np.add(tstx,ad1)

test_final = np.zeros((30,29))
test_final = np.transpose(tstx)
test_final = test_final - utr
test_final = test_final / norm

y8 = predict(test_final,parameters,layer,keep_prob)

print(y8)

#base = 'data/model/'
#state_vulnerability1 = base + 'blank_statet.csv'
state_vulnerability1 = 'blank_statet.csv'
state_data1 = pd.read_csv(state_vulnerability1)

state_data1.loc[:,['Vulnerability']]=y8[:]

if cap ==29:
    new_df1=state_data1
    print(new_df1)
else:
    m2 = cap
    #state_vulnerability2 = base + 'blank_statet1.csv'
    state_vulnerability2 = 'blank_statet1.csv'
    new_df1 = pd.read_csv(state_vulnerability2)
    new_df1.loc[m2:m2,['States','Vulnerability']] = state_data1.loc[m2:m2,['States','Vulnerability']]
    print("\n")
    print(new_df1.loc[m2:m2,['States','Vulnerability']])
    print("\n")

#state_geo = base + 'states2.json'
state_geo = 'states2.json'
geo_json_data = json.load(open(state_geo))

m = folium.Map(location=[21, 78], zoom_start=5)

folium.Choropleth(
    geo_data=geo_json_data,
    data=new_df1,
    columns=['States', 'Vulnerability'],
    key_on='feature.id',
    fill_color='YlOrRd',
    threshold_scale=[-1, 0, 1, 2, 3, 4],
    fill_opacity=0.7,
    line_opacity=0.7,
    legend_name='Vulnerability for year'+str(yar),
    reset=True,
    highlight=True

).add_to(m)


folium.LayerControl().add_to(m)


m.save(os.path.join('results', 'Prediction_for_'+sts+'_in_'+yar+'.html'))
m

print("Please check the RESULTS folder for your Visualised Result")
print("\n")