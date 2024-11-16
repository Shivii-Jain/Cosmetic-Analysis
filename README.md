# Cosmetic Product Analysis Web Application
This web application allows users to explore and analyze cosmetic products using various techniques such as T-SNE visualization, ingredient frequency analysis, and skin-type-based product recommendations. The app is built using Streamlit, a Python framework for creating interactive web applications, and incorporates various data analysis and visualization libraries like Pandas, Scikit-learn, Matplotlib, and Bokeh.
How to Run the Application Locally
To run this application on your local machine, follow these steps:

Clone this repository:

bash
Copy code
git clone https://github.com/yourusername/cosmetic-product-analysis.git
Navigate to the project directory:

bash
Copy code
cd cosmetic-product-analysis
Install the required dependencies:

Copy code
pip install -r requirements.txt
Download the dataset cosmetics.csv and place it in the project directory. The dataset should contain information about various cosmetic products, including columns like Name, Brand, Price, Rank, Ingredients, and skin type suitability (Dry, Oily, Sensitive).

Run the application using the Streamlit command:

arduino
Copy code
streamlit run app.py
Open a browser and go to the following URL:

arduino
Copy code
http://localhost:8501
