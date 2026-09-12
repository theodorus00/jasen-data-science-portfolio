import streamlit as st
import pandas as pd
from pathlib import Path



@st.cache_resource
def get_artifact():
    from bank_churn.src.predict import load_artifact
    return load_artifact()

st.set_page_config(
    page_title = "Jasen | Portfolio",
    page_icon = "🀄",
    layout = "wide",
    initial_sidebar_state = "expanded",
)

tab_home, tab_ml, tab_eda = st.tabs([
    "🏠 Home",
    "🤖 ML Predictor", 
    "📈 EDA"
])

# ══════════════════════════════════════════════
# TAB 1 — HOME
# ══════════════════════════════════════════════
with tab_home:

    st.title("Hi, I'm Jasen Theodorus 👋")
    st.subheader("Data Scientist | AI Engineer | Consultant | Product & Analytics")

    col_photo, col_bio = st.columns([1, 3])

    with col_photo:
        photo_path = Path(__file__).resolve().parent / "foto_jasen.jpg"
        if photo_path.exists():
            st.image(str(photo_path), width=200)
        else:
            st.info("Add foto_jasen.jpg to the app folder to display your photo.")

    with col_bio:
        st.write("""
        I am a Product & Analytics professional with more than three years of experience in the banking industry, with a growing specialization and strong interest in Data Science and Artificial Intelligence. With an academic background in Actuarial Science, I have built a strong foundation in mathematics, statistics, probability, and analytical problem-solving, which I combine with my professional experience at PT Bank Central Asia Tbk (BCA).

        My experience lies at the intersection of business, technology, data, and decision-making. Throughout my career, I have been involved in the development of various products and digital payment solutions while collaborating with cross-functional teams to solve both business and operational challenges. Several initiatives I have contributed to have delivered measurable impact, ranging from increased product adoption and improved operational efficiency to reducing recurring operational issues.

        To complement my professional experience, I have developed hands-on capabilities in Python, SQL, Machine Learning, statistical analysis, data processing, and Business Intelligence. I am accustomed to approaching problems end-to-end—from understanding business challenges, processing and analyzing data, developing and evaluating models, to translating analytical findings into actionable insights, business recommendations, and technical requirements that can be implemented.

        Through various projects, I have explored areas including customer behavior analysis, fraud detection, predictive analytics, classification, regression, customer segmentation, recommendation systems, as well as Generative AI and LLM-based applications.

        I believe that a strong Data and AI solution is not defined solely by the complexity of the model behind it, but by how effectively it solves real-world problems. I am therefore looking to further develop my career across Data Science, Artificial Intelligence, and Technology Consulting, where I can combine critical thinking, quantitative expertise, technology, business understanding, and collaboration to transform complex challenges into simple, practical, and impactful solutions.
        """)
        st.write("**Core Skills:** Communication Skills · EDA · ML Modeling · A/B Testing · LLM · RAG")

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Years of Experience", "3+")
    col2.metric("Industry", "Banking")
    col3.metric("👨‍🎓 Major", "Actuarial Science")
    col4.metric("Projects", "50+")

    st.divider()

    st.subheader("Featured Projects")

    p1, p2, p3 = st.columns(3)

    with p1:
        with st.container(border=True):
            st.markdown("### 🔽 Bank Churn Prediction")
            st.write("""
            The Payment Division of an online lending company wants to predict 
            which customers are likely to churn (become inactive customers).
            """)
            st.write(
                "**Test Precision = 84.53%** · "
                "**Test Recall = 72.31%** · "
                "**Test F1-Score = 77.94%**"
            )
            st.info("👉 Try it live in the **🤖 ML Predictor** tab above or **📈 EDA** tab above.")

    with p2:
        with st.container(border=True):
            st.markdown("### 📈 Google Stock Price with RNN")
            st.write("""
            ML Model trained to predict the stock price of GOOGL with 3 models. 
            simple LSTM, Deep LSTM + GRU, and BiLSTM
            """)
            st.write("**R² = 83%** · **MAPE = 2.22%**")
            st.info("👉 Try it live in the **🤖 ML Predictor** tab above or **📈 EDA** tab above.")
    
    with p3:
        with st.container(border=True):
            st.markdown("### 🏠 House Price Prediction")
            st.write("Estimate house sale prices using property features, notebook preprocessing, and regularized regression.")
            st.write("**Final model: ElasticNet** · **Evaluation: 5-fold CV on log-price**")
            st.info("Explore model results and CSV predictions in ML Predictor, or housing patterns in EDA.")

# ══════════════════════════════════════════════
# TAB 2 — ML PREDICTOR
# ══════════════════════════════════════════════
def show_ml():
    st.header("🔽 Bank Churn Prediction")

    st.write(
        "This project predicts potential customer churn "
        "to build retention candidate"
    )

    st.markdown(
        "**Chosen Model:** XGBoost  \n"
        "**Compared Model:** Logistic Regression, "
        "Decision Tree, dan Random Forest."
    )

    try:
        from bank_churn.src.predict import predict_customers
        artifact = get_artifact()
    except (FileNotFoundError, ImportError):
        st.error(
            "Model not defined. Please run python -m src.train."
        )
        return

    raw_columns = artifact["raw_feature_columns"]

    with st.expander("Format data input"):
        st.write(
            "Upload a CSV with the following columns. Use text "
            "categories as in the original dataset, before preprocessing."
        )

        st.dataframe(
            pd.DataFrame({"Kolom": raw_columns}),
            hide_index=True,
        )

        st.caption(
            "user_id can be included as an identifier. "
            "attrition_flag is not required for prediction."
        )

        st.download_button(
            "Download template CSV",
            data=pd.DataFrame(
                columns=raw_columns
            ).to_csv(index=False).encode("utf-8"),
            file_name="customer_input_template.csv",
            mime="text/csv",
        )

    uploaded_file = st.file_uploader(
        "Upload CSV customer",
        type=["csv"],
    )

    if uploaded_file is None:
        st.info("Upload data to see predict result.")
        return

    try:
        customers = pd.read_csv(uploaded_file)

        st.subheader("Preview input")
        st.dataframe(customers.head(10), hide_index=True)

        with st.spinner("Memproses prediksi..."):
            predictions = predict_customers(
                customers,
                artifact,
            )

    except (ValueError, TypeError, KeyError) as error:
        st.error(f"Data belum bisa diproses: {error}")
        return

    total = len(predictions)
    churn_count = int(predictions["prediction"].sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Total customer", f"{total:,}")
    col2.metric("Churn potential", f"{churn_count:,}")
    col3.metric(
        "Not a potential churn customer",
        f"{total - churn_count:,}",
    )

    output = customers.copy()

    output["model_prediction"] = (
        predictions["prediction"].to_numpy()
    )
    output["model_churn_probability"] = (
        predictions["churn_probability"].to_numpy()
    )
    output["model_prediction_label"] = (
        predictions["prediction_label"].to_numpy()
    )

    st.subheader("Hasil prediksi")
    st.dataframe(output, hide_index=True)

    st.caption(
        "Probabilities range are from 0 to 1. "
        "Probability = 0,25 , then churn estimation probabilities churn 25%. "
        "Predictions are model estimates, not guarantees."
    )

    st.download_button(
        "Download prediction result",
        data=output.to_csv(index=False).encode("utf-8-sig"),
        file_name="customer_churn_predictions.csv",
        mime="text/csv",
    )

def show_rnn_history():
    st.header("Google Stock Price — RNN/LSTM")
    st.write(
        "Historical experiment using GOOGL closing prices. "
        "Training: 2010–2018. Validation: 2019."
    )
    st.caption(
        "These are saved outputs from the original notebook. "
        "No model is retrained on this page."
    )

    reports_path = (
        Path(__file__).resolve().parent
        / "google_stock_price"
        / "reports"
        / "historical"
    )

    comparison_path = reports_path / "comparison.txt"

    if not comparison_path.exists():
        st.info(
            "Extract the historical outputs into "
            "google_stock_price/reports/historical first."
        )
        return

    st.subheader("Original model comparison")
    st.code(
        comparison_path.read_text(encoding="utf-8"),
        language=None,
    )

    model_options = {
        "Simple LSTM": "model_1",
        "Deep LSTM + GRU": "model_2",
        "Bidirectional LSTM": "model_3",
    }

    selected_model = st.selectbox(
        "Select a model to inspect",
        list(model_options),
        key="rnn_history_model",
    )

    prefix = model_options[selected_model]

    for title, suffix in [
        ("Training and validation loss", "loss"),
        ("Historical predictions vs actual prices", "predictions"),
    ]:
        st.subheader(title)
        image_path = reports_path / f"{prefix}_{suffix}.png"

        if image_path.exists():
            st.image(str(image_path))
        else:
            st.info(f"Image unavailable: {image_path.name}")

    st.caption(
        "Original notebook limitation: Model 3 uses 222 validation "
        "observations, while Models 1 and 2 use 252. "
        "The original Model 3 plot also has misaligned date labels. "
        "The historical outputs are displayed unchanged."
    )

# ══════════════════════════════════════════════
# TAB 3 — EDA DASHBOARD
# ══════════════════════════════════════════════
def show_eda():
    st.header("Exploratory Data Analysis")
    st.write(
        "Eksplorasi karakteristik customer dan distribusi churn "
        "menggunakan data training sebelum preprocessing."
    )

    try:
        from bank_churn.src.data import load_data, split_data
        df = load_data()
        X_train, _, _, y_train, _, _ = split_data(df)
    except (OSError, ValueError, KeyError, ImportError) as error:
        st.error(f"Dataset EDA belum bisa dibaca: {error}")
        return

    # Salinan untuk visualisasi; tidak mengubah data model.
    eda = X_train.copy()
    eda["churn"] = y_train
    eda["status"] = eda["churn"].map(
        {0: "Tidak Churn", 1: "Churn"}
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Customer training", f"{len(eda):,}")
    col2.metric("Fitur mentah", X_train.shape[1])
    col3.metric("Churn rate training", f"{eda['churn'].mean():.2%}")

    with st.expander("Lihat data dan statistik"):
        st.dataframe(eda.head(20), hide_index=True)
        st.dataframe(
            X_train.describe(include="all").fillna("").astype(str)
        )

        st.markdown("**Jumlah nilai kosong per fitur**")
        st.dataframe(
            X_train.isna().sum().rename("missing_count")
        )

    st.subheader("Distribusi churn")

    distribution = (
        eda["status"]
        .value_counts()
        .rename("Jumlah customer")
    )
    st.bar_chart(distribution)

    st.subheader("Churn berdasarkan kategori")

    category_labels = {
        "Gender": "gender",
        "Pendidikan": "education_level",
        "Pendapatan": "income_category",
        "Status pernikahan": "marital_status",
        "Kategori kartu": "card_category",
        "Jumlah kontak dalam 12 bulan": "contacts_count_12_mon",
    }

    selected_label = st.selectbox(
        "Pilih karakteristik customer",
        list(category_labels),
    )

    feature = category_labels[selected_label]

    summary = (
        eda.groupby(feature, observed=True, dropna=False)["churn"]
        .agg(
            jumlah_customer="count",
            jumlah_churn="sum",
            churn_rate="mean",
        )
    )

    summary["churn_rate_pct"] = summary["churn_rate"] * 100
    summary = summary.drop(columns="churn_rate")
    summary.index = summary.index.map(str)

    left, right = st.columns(2)

    with left:
        st.markdown("**Jumlah customer per kelompok**")
        st.bar_chart(summary[["jumlah_customer"]])

    with right:
        st.markdown("**Churn rate per kelompok (%)**")
        st.bar_chart(summary[["churn_rate_pct"]])

    st.dataframe(summary.round(2))

    st.caption(
        "Churn rate adalah jumlah customer churn dibagi total "
        "customer dalam kelompok. Perhatikan ukuran kelompok "
        "saat membandingkan persentasenya."
    )

    st.subheader("Distribusi usia dan lama menjadi nasabah")

    numeric_labels = {
        "Usia customer": "customer_age",
        "Lama menjadi nasabah (bulan)": "months_on_book",
    }

    numeric_label = st.selectbox(
        "Pilih variabel",
        list(numeric_labels),
    )

    numeric_feature = numeric_labels[numeric_label]

    distribution_by_status = pd.crosstab(
        eda[numeric_feature],
        eda["status"],
    )

    st.bar_chart(distribution_by_status)

    st.caption(
        "Grafik menunjukkan jumlah customer pada setiap nilai "
        "variabel, dipisahkan berdasarkan status churn. "
        "Hubungan yang terlihat tidak membuktikan sebab-akibat."
    )

def show_rnn_eda():
    st.header("EDA — Google Stock Price")

    st.write(
        "This project uses GOOGL daily closing prices "
        "from 2010 to 2019."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Ticker", "GOOGL")
    col2.metric("Training period", "2010–2018")
    col3.metric("Validation period", "2019")

    st.subheader("Historical prices and chronological split")

    image_path = (
        Path(__file__).resolve().parent
        / "google_stock_price"
        / "reports"
        / "historical"
        / "train_validation_split.png"
    )

    if image_path.exists():
        st.image(str(image_path))
    else:
        st.info(
            "The historical chart is missing. Check "
            "google_stock_price/reports/historical/train_validation_split.png."
        )

    st.write(
        "The dataset is split chronologically: earlier observations "
        "are used for training, and 2019 observations are used "
        "for validation."
    )

    st.caption(
        "This is the original chart saved in the notebook. "
        "It displays historical data, not live market prices."
    )

def show_house_price_pred():
    from house_price_prediction.portfolio import show_house_ml
    show_house_ml()


with tab_ml:
    selected_project = st.selectbox(
        "Select a project",
        [
            "Bank Churn Prediction",
            "Google Stock — RNN/LSTM",
            "House Price Prediction"
        ],
        key="ml_project_selector",
    )

    if selected_project == "Bank Churn Prediction":
        show_ml()
    elif selected_project == "Google Stock — RNN/LSTM":
        show_rnn_history()
    else:
        show_house_price_pred()

with tab_eda:
    selected_eda_project = st.selectbox(
        "Select a project",
        [
            "Bank Churn Prediction",
            "Google Stock — RNN/LSTM",
            "House Price Prediction"
        ],
        key="eda_project_selector",
    )

    if selected_eda_project == "Bank Churn Prediction":
        show_eda()
    elif selected_eda_project == "Google Stock — RNN/LSTM":
        show_rnn_eda()
    else:
        from house_price_prediction.portfolio import show_house_eda
        show_house_eda()