import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.title("Customer Segmentation App")
st.write ("Group Customers for better marketing")

df = pd.read_csv("Mall_Customers(1).csv")

k = st.slider("How many customer groups?", 2, 6, 3)

X = df[['Annual Income (k$)','Spending Score (1-100']]
kmeans = KMeans (n_clusters=k, random_state=42)
df['Group'] = kmeans.fit_predict(X)

st.write(f"showing{k} groups:")
st.dataframe(df.head())

fig, ax = plt.subplots()
for group in range(k):
    d = df[df['Group'] == group]
    ax.scatter(d['Annual Income (k$)'], d['Spending Score (1-100'], label = f"Group{group}")
ax.set_xlabel("Income")
ax.set_ylabel("Spending Score")
ax.legend()
st.pyplot(fig)

csv = df.to_csv(index=False)
st.download_button ("Download Results", csv, "segmented.csv", "text/csv")
