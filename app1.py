# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer               
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso

# Load the training data
train_path = 'filtered_data.csv'
train_data = pd.read_csv(train_path)

# Remove rows with missing SalePrice values and drop the Id column
train_data.dropna(subset=['SalePrice'], inplace=True)
train_data = train_data.drop(columns='Id')

# Define the features used in prediction and correlation with SalePrice
features_used = [
    'OverallQual', 'GarageCars', 'FullBath', 'YearBuilt',
    'GarageArea', 'TotalBsmtSF', 'GrLivArea', 'KitchenQual',
    'ExterQual', 'CentralAir', 'GarageType', 'MSZoning'
]
X = train_data[features_used]
y = train_data['SalePrice']

# Preprocessing for numerical and categorical features
numeric_features = ['OverallQual', 'GarageCars', 'FullBath', 'YearBuilt', 'GarageArea', 'TotalBsmtSF', 'GrLivArea']
categorical_features = ['KitchenQual', 'ExterQual', 'CentralAir', 'GarageType', 'MSZoning']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Define the Lasso regression model
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Lasso(alpha=0.001, random_state=42))
])

# Train the model on the entire training dataset
model_pipeline.fit(X, y)

# Visualizations
st.title('Dự đoán giá nhà với các đặc trưng được sử dụng trong dự đoán')

# Select plot type
plot_option = st.selectbox(
    'Chọn loại sơ đồ để hiển thị:',
    ['Biểu đồ phân tán cho biến số', 'Biểu đồ hộp cho biến phân loại', 'Biểu đồ giá bán', 'Heatmap của tất cả các biến']
)

# Display selected plot with comments
if plot_option == 'Biểu đồ phân tán cho biến số':
    st.subheader('Biểu đồ phân tán các đặc trưng số so với SalePrice')
    fig, axes = plt.subplots(nrows=len(numeric_features), ncols=1, figsize=(8, 3 * len(numeric_features)))
    for i, feature in enumerate(numeric_features):
        sns.scatterplot(data=train_data, x=feature, y='SalePrice', ax=axes[i])
        axes[i].set_title(f'{feature} vs SalePrice')
    plt.tight_layout()
    st.pyplot(fig)

elif plot_option == 'Biểu đồ hộp cho biến phân loại':
    st.subheader('Biểu đồ hộp cho các đặc trưng phân loại')
    fig, axes = plt.subplots(nrows=len(categorical_features), ncols=1, figsize=(8, 3 * len(categorical_features)))
    for i, feature in enumerate(categorical_features):
        sns.boxplot(x=train_data[feature], y=train_data['SalePrice'], ax=axes[i])
        axes[i].set_title(f'{feature} vs SalePrice')
    plt.tight_layout()
    st.pyplot(fig)

elif plot_option == 'Biểu đồ giá bán':
    st.subheader('Biểu đồ phân bố SalePrice')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(train_data['SalePrice'], kde=True, color='blue')
    ax.set_title('Phân bố giá bán nhà')
    ax.set_xlabel('SalePrice')
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    st.pyplot(fig)

elif plot_option == 'Heatmap của tất cả các biến':
    st.subheader('Heatmap tương quan giữa tất cả các biến')
    fig, ax = plt.subplots(figsize=(12, 8))
    correlation_matrix = train_data[features_used + ['SalePrice']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax, fmt=".2f", linewidths=.5)
    st.pyplot(fig)

# Sidebar with form inputs for feature values
st.sidebar.subheader('Dự đoán giá nhà lý tưởng của bạn')
st.sidebar.markdown('Trả lời các câu hỏi sau về ngôi nhà lý tưởng của bạn trong thị trường bất động sản của Ames để biết giá của nó.')
with st.sidebar.form(key='house_form'):
    # Numerical features input
    OverallQual = st.slider('Chất lượng tổng thể:', 1, 10, value=int(X['OverallQual'].median()))
    GarageCars = st.slider('Sức chứa của garage:', 0, 4, value=int(X['GarageCars'].median()))
    FullBath = st.slider('Số lượng phòng tắm đầy đủ:', 0, 3, value=int(X['FullBath'].median()))
    YearBuilt = st.slider('Năm xây dựng:', 1872, 2010, value=int(X['YearBuilt'].median()))
    GarageArea = st.slider('Diện tích garage (mét vuông):', 0, 1500, value=int(X['GarageArea'].median()))
    TotalBsmtSF = st.slider('Tổng diện tích tầng hầm (mét vuông):', 0, 5000, value=int(X['TotalBsmtSF'].median()))
    GrLivArea = st.slider('Diện tích sử dụng trên mặt đất (mét vuông):', 0, 5000, value=int(X['GrLivArea'].median()))

    # Categorical features input
    KitchenQual = st.selectbox('Chất lượng nhà bếp:', X['KitchenQual'].unique())
    ExterQual = st.selectbox('Chất lượng vật liệu ngoại thất:', X['ExterQual'].unique())
    CentralAir = st.selectbox('Có điều hòa trung tâm không?', ['Y', 'N'])
    GarageType = st.selectbox('Kiểu garage:', X['GarageType'].unique())
    MSZoning = st.selectbox('Vùng dân cư:', X['MSZoning'].unique())

    submit_button = st.form_submit_button(label='Dự đoán')

# Process the form submission
if submit_button:
    # Prepare input data for prediction
    input_data = pd.DataFrame([{
        'OverallQual': OverallQual, 'GarageCars': GarageCars, 'FullBath': FullBath,
        'YearBuilt': YearBuilt, 'GarageArea': GarageArea, 'TotalBsmtSF': TotalBsmtSF,
        'GrLivArea': GrLivArea, 'KitchenQual': KitchenQual, 'ExterQual': ExterQual,
        'CentralAir': CentralAir, 'GarageType': GarageType, 'MSZoning': MSZoning
    }])
    # Process features as needed before prediction
    processed_data = model_pipeline.named_steps['preprocessor'].transform(input_data)
    predicted_price = model_pipeline.named_steps['regressor'].predict(processed_data)[0]

    st.write('Giá nhà dự đoán:', f'${predicted_price:,.2f}')


