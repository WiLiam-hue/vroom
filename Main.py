import streamlit as st
import pandas as pd
import math
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Used Car Price Prediction System", layout='wide')

st.title("Vehicle Dashboard :car:")

# define Euler's number
euler_number = math.e

df = pd.read_csv('car_data.csv')

# Create a sidebar
st.sidebar.title('Filters')

brand_types = st.sidebar.multiselect(
    'Filter by Car Brand', df['brand'].unique())
# filter the brand_types in df using the selected brand types
if brand_types:
    df = df[df['brand'].isin(brand_types)]

vehicle_types = st.sidebar.multiselect(
    'Filter by Vehicle Type', df['vehicleType'].unique())

if vehicle_types:
    df = df[df['vehicleType'].isin(vehicle_types)]

# Create filter for car model
model_types = st.sidebar.multiselect(
    'Filter by Model Type', df['model'].unique())

# Filter the df based on the selected model
if model_types:
    df = df[df['model'].isin(model_types)]

# Create filter for gearbox
gb_types = st.sidebar.multiselect(
    'Filter by Gearbox Type', df['gearbox'].unique())

# Filter the df based on the selected gearbox
if gb_types:
    df = df[df['gearbox'].isin(gb_types)]

# Create filter for fuelType
fuel_types = st.sidebar.multiselect(
    'Filter by Fuel Type', df['fuelType'].unique())

# Filter the df based on the selected fuel types
if fuel_types:
    df = df[df['fuelType'].isin(fuel_types)]

repair_types = st.sidebar.selectbox('Filter by Repair Type', options=[
    'All'] + df['notRepairedDamage'].unique().tolist())
if repair_types == 'All':
    df = df
else:
    df = df[df['notRepairedDamage'] == repair_types]

# Define the slider for carAge
min_age = 4
max_age = 36
age_range = st.sidebar.slider('Select Car Age', min_age,
                              max_age, (min_age, max_age))

# Filter the DataFrame based on the selected car age range
df = df[(df['carAge'] >= age_range[0]) & (df['carAge'] <= age_range[1])]

# Define the slider for kilometer
min_kilo = 5000
max_kilo = 150000
kilo_range = st.sidebar.slider('Select Travelled Milleage (KM)',
                               min_kilo, max_kilo, (min_kilo, max_kilo))

# Filter the DataFrame based on the selected car kilo range
df = df[(df['kilometer'] >= kilo_range[0]) &
        (df['kilometer'] <= kilo_range[1])]

# Perform log transformation on 'powerPS' column
constant = df['powerPS_log'] * np.log10(math.e)
df['powerPS1'] = 10 ** constant

# Define the slider for powerPS
min_power = 36
max_power = 312
power_range = st.sidebar.slider(
    'Select Car Power', min_power, max_power, (min_power, max_power))

# Filter the DataFrame based on the selected powerPS range
df = df[(df['powerPS1'] >= power_range[0]) &
        (df['powerPS1'] <= power_range[1])]

col1, col2, col3 = st.columns(3)
with col1:
    col1.metric(label="Number of Record", value=len(df))

with col2:
    col2.metric(label="Average Car Price", value=round(df['price'].mean(), 2))

with col3:
    col3.metric(label="Median Car Price", value=round(df['price'].median(), 2))

# Change the data type of the "yearOfRegistration" column to object
df['yearOfRegistration'] = df['yearOfRegistration'].astype(str)

# Sort the DataFrame by the "yearOfRegistration" column in ascending order
df = df.sort_values('yearOfRegistration')

# Create the line plot using seaborn
plt.figure(figsize=(16, 3))  # Increase the width of the plot
sns.lineplot(data=df, x='yearOfRegistration', y='price')
plt.title('Year of Registration vs Price')
plt.xlabel('Year of Registration')
plt.ylabel('Price')

# Remove the border
sns.despine()

# Rotate the x-axis labels by 90 degrees
plt.xticks(rotation=90)
# Decrease the font size of the x-axis labels
plt.xticks(fontsize=5)

# Show the plot using Streamlit
st.pyplot(plt)

col1, col2 = st.columns(2)
with col1:

    # Value count for vehicleType
    vehicle_counts = df['vehicleType'].value_counts(sort=True, ascending=False)

    # Check if vehicle_counts is empty
    if vehicle_counts.empty:
        st.write("No records found in Vehicle Count.")
    else:
        # Create a bar chart using Plotly Express
        fig = px.bar(vehicle_counts, x=vehicle_counts.index,
                     y=vehicle_counts.values)

        # Set the axis labels and chart title
        fig.update_xaxes(title='Vehicle Type')
        fig.update_yaxes(title='Count')
        fig.update_layout(title='Vehicle Type Counts', width=500, height=300)

        # Display the chart
        st.plotly_chart(fig)

        # Value count for brand
    brand_counts = df['brand'].value_counts().head(10)

    # Check if brand_counts is empty
    if brand_counts.empty:
        st.write("No records found for the top ten car brands.")
    else:
        # Create a bar chart using Plotly Express
        fig = px.bar(brand_counts, x=brand_counts.index, y=brand_counts.values)

        # Set the axis labels and chart title
        fig.update_xaxes(title='Car Brand')
        fig.update_yaxes(title='Count')
        fig.update_layout(title='Top Ten Car Brands', width=500, height=300)

        # Display the chart
        st.plotly_chart(fig)

    # Count the occurrences of each value in the notRepairedDamage
    repair_counts = df['notRepairedDamage'].value_counts()

    if repair_counts.empty:
        st.write("No records found in Repair Count.")
    else:
        # Create a pie chart
        fig = px.pie(repair_counts, values=repair_counts.values, names=repair_counts.index,
                     title='Pie Chart of Repair Type', width=500,
                     height=300)

        # Display the chart
        st.plotly_chart(fig)

    # Count the occurrences of each fuel type
    f_counts = df['fuelType'].value_counts(sort=True, ascending=True)

    # Create an if else condition for fuel type
    if f_counts.empty:
        st.write("No records found in Fuel Count.")
    else:
        # Create a horizontal bar chart of the sorted vehicle counts using Plotly Express
        fig = px.bar(f_counts, y=f_counts.index,
                     x=f_counts.values, orientation='h')

        # Set the axis labels and chart title
        fig.update_xaxes(title='Count')
        fig.update_yaxes(title='Fuel Type')
        fig.update_layout(title='Fuel Type Counts', width=500,
                          height=300)  # Set the size of the chart

        # Show the chart
        st.plotly_chart(fig)

with col2:

    # Calculate the average price for each vehicle type
    average_price_vehicle = df.groupby(
        'vehicleType')['price'].mean().reset_index()

    if average_price_vehicle.empty:
        st.write("No record found in Average price for each Vehicle Type.")
    else:
        # Sort the DataFrame by average price in descending order
        average_price_vehicle = average_price_vehicle.sort_values(
            'price', ascending=False)

        # Create the bar chart using Plotly Express
        fig = px.bar(average_price_vehicle, x='vehicleType', y='price')

        # Set the axis labels and chart title
        fig.update_xaxes(title='Vehicle Type')
        fig.update_yaxes(title='Average Price')
        fig.update_layout(title='Average Price by Vehicle Type',
                          width=500, height=300)

        # Show the chart using Streamlit
        st.plotly_chart(fig)

        # Calculate the average price for each brand
    average_price_brand = df.groupby('brand')['price'].mean().reset_index()

    # Sort the DataFrame by average price in descending order
    average_price_brand = average_price_brand.sort_values(
        'price', ascending=False).head(10)

    if average_price_brand.empty:
        st.write("No record found in Average Price for the top ten car brands.")
    else:
        # Create the bar chart using Plotly Express
        fig = px.bar(average_price_brand, x='brand', y='price')

        # Set the axis labels and chart title
        fig.update_xaxes(title='Car Brand')
        fig.update_yaxes(title='Average Price')
        fig.update_layout(
            title='Average Price for Top Ten Car Brands', width=500, height=300)

        # Show the chart using Streamlit
        st.plotly_chart(fig)

        # Calculate the average price for each Repair type
    average_price_repair = df.groupby('notRepairedDamage')[
        'price'].mean().reset_index()

    if average_price_repair.empty:
        st.write("No record found in average price for each Repair Type.")
    else:
        # Sort the DataFrame by average price in descending order
        average_price_repair = average_price_repair.sort_values(
            'price', ascending=False)

        # Create the bar chart using Plotly Express
        fig = px.bar(average_price_repair, x='notRepairedDamage', y='price')

        # Set the axis labels and chart title
        fig.update_xaxes(title='Repair Type')
        fig.update_yaxes(title='Average Price')
        fig.update_layout(title='Average Price by Repair Type',
                          width=500, height=300)

        # Show the chart using Streamlit
        st.plotly_chart(fig)

    # Calculate the average price for each vehicle type
    average_price = df.groupby('fuelType')['price'].mean().reset_index()

    if average_price.empty:
        st.write("No record found in Average price for each Fuel Type.")
    else:
        # Sort the DataFrame by average price in descending order
        average_price = average_price.sort_values('price', ascending=False)

        # Create the bar chart using Plotly Express
        fig = px.bar(average_price, x='fuelType', y='price')

        # Set the axis labels and chart title
        fig.update_xaxes(title='Fuel Type')
        fig.update_yaxes(title='Average Price')
        fig.update_layout(title='Average Price by Fuel Type',
                          width=500, height=300)

        # Show the chart using Streamlit
        st.plotly_chart(fig)
