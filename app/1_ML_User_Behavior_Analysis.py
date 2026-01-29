import streamlit as st

st.set_page_config(
    page_title="ML User Behavior Analysis",
    page_icon="📊",
    layout="wide"
)

st.sidebar.title("📌 Navigation")
st.sidebar.markdown(
    """
    This menu allows navigation between different stages of the
    Machine Learning pipeline, including model training,
    evaluation, comparison, and result interpretation.
    """
)


st.sidebar.info(
    """
    🔧 Technologies:
    - Python
    - NumPy
    - pandas
    - scikit-learn
    - imbalanced-learn
    - Matplotlib
    - Seaborn
    - Streamlit
    """
)


st.title("📊 ML User Behavior Analysis")

st.markdown(
    """
    ### General overview

    This application demonstrates a complete Machine Learning workflow
    for **user behavior analysis** in a simulated music platform scenario.

    The project uses **synthetic data** to model user activity patterns
    (listening behavior, social interaction, content engagement) and focuses on:

    - predicting whether a user is likely to become a premium subscriber
    - estimating a continuous engagement score
    - comparing multiple Machine Learning models and evaluation metrics
    - analyzing the impact of data imbalance and preprocessing techniques
    - interpreting model behavior using feature importance

    The goal of this application is **educational**: to showcase how
    classification, regression, evaluation, and explainability techniques
    can be combined into a coherent ML pipeline.

    ---
    Use the menu on the left to navigate between sections.
    """
)


