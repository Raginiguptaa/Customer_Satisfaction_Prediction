import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
data = pd.read_csv("C:\\Users\\welcome\\Desktop\\Customer_Satisfaction_Prediction\\DATASET\\customer_support_tickets.csv")
print(data.head())
print(data.info())
print(data.describe())
print(data.columns)
print(data.isnull().sum())
common_issues = data['Ticket Subject'].value_counts().head(10)
print("Top 10 Common Issues:")
print(common_issues)
data['Date of Purchase'] = pd.to_datetime(data['Date of Purchase'])
data['YearMonth'] = data['Date of Purchase'].dt.to_period('M')
ticket_trends = data.groupby('YearMonth').size()

plt.figure(figsize=(10, 6))
ticket_trends.plot(kind='line', marker='o')
plt.title('Customer Support Ticket Trends Over Time')
plt.xlabel('Year-Month')
plt.ylabel('Number of Tickets')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('C:\\Users\\welcome\\Desktop\\Customer_Satisfaction_Prediction\\OUTPUTS\\ticket_trends.PNG')
plt.show()
ticket_type_segmentation = data.groupby('Ticket Type').size()
print("\nSegmentation based on Ticket Types:")
print(ticket_type_segmentation)
satisfaction_segmentation = data.groupby('Customer Satisfaction Rating').size()
print("\nSegmentation based on Customer Satisfaction Levels:")
print(satisfaction_segmentation)

sns.set(style="whitegrid")

plt.figure(figsize=(10, 6)) #figsize=(10, 6) means:Width = 10 units ,Height = 6 units
sns.histplot(data['Customer Satisfaction Rating'], bins=5,
kde=True, color='skyblue') #bins=5 means:Divides the data into 5 intervals ,kde=True means: Kernel Density Estimate (Adds a smooth curve)
plt.title('Customer Satisfaction Distribution')
plt.xlabel('Satisfaction Rating')
plt.ylabel('Frequency')
plt.savefig('C:\\Users\\welcome\\Desktop\\Customer_Satisfaction_Prediction\\OUTPUTS\\customer_satisfaction_distribution.PNG')
plt.show()

ticket_status_distribution = data['Ticket Status'].value_counts() #Counts the number of tickets in each status category
plt.figure(figsize=(8, 8))
plt.pie(ticket_status_distribution,labels=ticket_status_distribution.index, autopct='%1.1f%%',colors=sns.color_palette('pastel'), startangle=140)
plt.title('Ticket Status Distribution')
plt.axis('equal')
plt.savefig('C:\\Users\\welcome\\Desktop\\Customer_Satisfaction_Prediction\\OUTPUTS\\ticket_status_distribution.PNG')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['Customer Age'], bins=20, kde=True, color='salmon') 
plt.title('Customer Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('C:\\Users\\welcome\\Desktop\\Customer_Satisfaction_Prediction\\OUTPUTS\\customer_age_distribution.PNG')
plt.show()

customer_gender_distribution = data['Customer Gender'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(customer_gender_distribution,
labels=customer_gender_distribution.index, autopct='%1.1f%%',
colors=sns.color_palette('Set2'), startangle=90)
plt.title('Customer Gender Distribution')
plt.axis('equal')
plt.savefig('C:\\Users\\welcome\\Desktop\\Customer_Satisfaction_Prediction\\OUTPUTS\\customer_gender_distribution.PNG')
plt.show()
plt.figure(figsize=(10, 6))
ticket_channel_distribution = data['Ticket Channel'].value_counts()
sns.barplot(
    x=ticket_channel_distribution.index,
    y=ticket_channel_distribution,
    hue=ticket_channel_distribution.index,
    palette='rocket',
    legend=False
)
plt.title('Ticket Channel Distribution')
plt.xlabel('Ticket Channel')
plt.ylabel('Count')
plt.xticks(rotation=45) #Rotates x-axis labels for better readability
plt.savefig('C:\\Users\\welcome\\Desktop\\Customer_Satisfaction_Prediction\\OUTPUTS\\ticket_channel_distribution.PNG')
plt.show()

average_satisfaction = data.groupby('Customer Gender')['Customer Satisfaction Rating'].mean().reset_index()

plt.figure(figsize=(8, 6))
sns.barplot(
    x='Customer Gender',
    y='Customer Satisfaction Rating',
    data=average_satisfaction,
    hue='Customer Gender',
    palette='muted',
    order=['Male', 'Female', 'Other'],
    legend=False
)
plt.title('Average Customer Satisfaction by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Satisfaction Rating')
plt.ylim(1, 5) # Adjust y-axis limit if needed
plt.savefig('C:\\Users\\welcome\\Desktop\\Customer_Satisfaction_Prediction\\OUTPUTS\\average_satisfaction_by_gender.PNG')
plt.show()

plt.figure(figsize=(10, 6))
product_purchased_distribution = data['Product Purchased'].value_counts().head(10)
sns.barplot(
    y=product_purchased_distribution.index,
    x=product_purchased_distribution,
    hue=product_purchased_distribution.index,
    palette='magma',
    legend=False
)

plt.title('Top 10 Products Purchased')
plt.xlabel('Count')
plt.ylabel('Product')
plt.savefig('C:\\Users\\welcome\\Desktop\\Customer_Satisfaction_Prediction\\OUTPUTS\\top_products.PNG')
plt.show()

# Chart 2: Top Items Purchased by Gender (Horizontal Bar Chart)
plt.figure(figsize=(15, 6))

# Top Items Purchased by Males
plt.subplot(1, 3, 1) #1 row, 3 columns, 1st plot
top_items_male = data[data['Customer Gender'] == 'Male']['Product Purchased'].value_counts().head(5)
top_items_male.plot(kind='barh', color='skyblue')
plt.title('Top Items Purchased by Males')

plt.xlabel('Count')
plt.ylabel('Product')

# Top Items Purchased by Females
plt.subplot(1, 3, 2) #1 row, 3 columns, 2nd plot
top_items_female = data[data['Customer Gender'] =='Female']['Product Purchased'].value_counts().head(5)
top_items_female.plot(kind='barh', color='salmon')
plt.title('Top Items Purchased by Females')
plt.xlabel('Count')
plt.ylabel('Product')

plt.subplot(1, 3, 3) #1 row, 3 columns, 3rd plot
top_items_other = data[data['Customer Gender'] =='Other']['Product Purchased'].value_counts().head(5)
top_items_other.plot(kind='barh', color='lightgreen')
plt.title('Top Items Purchased by Other Genders')
plt.xlabel('Count')
plt.ylabel('Product')

plt.tight_layout()
plt.savefig('C:\\Users\\welcome\\Desktop\\Customer_Satisfaction_Prediction\\OUTPUTS\\top_items_by_gender.PNG')
plt.show()

# Count ticket types
ticket_type_distribution = data['Ticket Type'].value_counts()

# Plot
plt.figure(figsize=(8, 6))
ticket_type_distribution.plot(kind='pie', autopct='%1.1f%%',
colors=['skyblue', 'salmon', 'lightgreen'])
plt.title('Ticket Type Distribution')
plt.ylabel('')
plt.savefig('C:\\Users\\welcome\\Desktop\\Customer_Satisfaction_Prediction\\OUTPUTS\\ticket_type_distribution.PNG')
plt.show()

# Count ticket priorities
priority_distribution = data['Ticket Priority'].value_counts()

# Plot
plt.figure(figsize=(8, 6))
priority_distribution.plot(kind='pie', autopct='%1.1f%%',
colors=['lightblue', 'lightgreen', 'lightsalmon', 'skyblue'])
plt.title('Priority Level Distribution')
plt.ylabel('')
plt.savefig('C:\\Users\\welcome\\Desktop\\Customer_Satisfaction_Prediction\\OUTPUTS\\priority_distribution.PNG')
plt.show()

bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61-70',
'71-80', '81-90', '91-100']

data['Age Group'] = pd.cut(data['Customer Age'], bins=bins,
labels=labels, right=False)

tickets_by_age_group = data.groupby('Age Group').size()

plt.figure(figsize=(10, 6))
tickets_by_age_group.plot(kind='bar', color='skyblue')
plt.title('Tickets Raised by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Number of Tickets Raised')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.savefig('C:\\Users\\welcome\\Desktop\\Customer_Satisfaction_Prediction\\OUTPUTS\\tickets_by_age_group.PNG')
plt.show()

data.replace([np.inf, -np.inf], np.nan, inplace=True) #Cleansing step to handle infinite values

g = sns.FacetGrid(data, col='Ticket Type', col_wrap=3,
height=5, aspect=1.5)#Creates a grid of plots ,col='Ticket Type' means:Separate plot for each ticket type ,col_wrap=3 means:Wraps plots into rows of 3
g.map(sns.histplot, 'Customer Age', bins=20, kde=True)

g.set_titles('{col_name}')
g.set_axis_labels('Age', 'Number of Tickets')

plt.subplots_adjust(top=0.9)
g.fig.suptitle('Distribution of Ticket Types by Age')
plt.savefig('C:\\Users\\welcome\\Desktop\\Customer_Satisfaction_Prediction\\OUTPUTS\\facet_grid.PNG')
plt.show()

plt.figure(figsize=(20, 14))
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

sns.histplot(
    data['Customer Satisfaction Rating'],
    bins=5,
    kde=True,
    color='skyblue',
    ax=axes[0, 0]
)
axes[0, 0].set_title('Customer Satisfaction Distribution')
axes[0, 0].set_xlabel('Satisfaction Rating')
axes[0, 0].set_ylabel('Frequency')

ticket_status = data['Ticket Status'].value_counts()

axes[0, 1].pie(
    ticket_status,
    labels=ticket_status.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=sns.color_palette('pastel')
)
axes[0, 1].set_title('Ticket Status Distribution')
axes[0, 1].axis('equal')

gender_dist = data['Customer Gender'].value_counts()

axes[1, 0].pie(
    gender_dist,
    labels=gender_dist.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=sns.color_palette('Set2')
)
axes[1, 0].set_title('Customer Gender Distribution')
axes[1, 0].axis('equal')

sns.boxplot(
    data=data,
    x='Ticket Type',
    y='Customer Age',
    ax=axes[1, 1]
)
axes[1, 1].set_title('Ticket Type vs Customer Age')
axes[1, 1].set_xlabel('Ticket Type')
axes[1, 1].set_ylabel('Customer Age')

plt.suptitle(
    'Customer Support Analytics Dashboard',
    fontsize=20,
    fontweight='bold'
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('C:\\Users\\welcome\\Desktop\\Customer_Satisfaction_Prediction\\OUTPUTS\\dashboard.PNG')
plt.show()
