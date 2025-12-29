import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import io

# إعداد الصفحة
st.set_page_config(
    page_title="تحليل أسعار لاعبي كرة القدم",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# تخصيص التصميم باستخدام CSS (High-Fidelity Sapphire Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&family=Noto+Sans+Arabic:wght@400;700;800&display=swap');

    /* المظهر العام - أزرق عميق وتقني */
    .stApp {
        background-color: #020617;
        color: #f8fafc;
        font-family: 'Noto Sans Arabic', 'Inter', sans-serif;
    }
    
    .main {
        background-color: #020617;
    }

    /* العنوان الرئيسي - ضخم ومشع */
    h1 {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 3.5rem !important;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: -2px;
        margin-bottom: 3rem !important;
        text-shadow: 0 0 30px rgba(59, 130, 246, 0.4);
    }
    
    /* العناوين الجانبية */
    h5 {
        color: #3b82f6 !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1.5rem !important;
        border-right: 5px solid #3b82f6;
        padding-right: 15px;
        letter-spacing: 0.05em;
    }

    /* الزر السحري - توهج سافاير */
    .stButton>button {
        width: 100%;
        background: linear-gradient(180deg, #3b82f6 0%, #2563eb 100%);
        color: #ffffff !important;
        font-size: 22px !important;
        font-weight: 800 !important;
        border-radius: 12px;
        padding: 20px;
        border: 2px solid #60a5fa;
        box-shadow: 0 0 25px rgba(59, 130, 246, 0.5);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase;
    }

    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 40px rgba(59, 130, 246, 0.8);
        border-color: #ffffff;
    }

    /* الخانات - تصميم تقني مظلم */
    div[data-baseweb="select"], div[data-baseweb="input"], .stNumberInput input, .stSelectbox div {
        background-color: #0f172a !important;
        color: #ffffff !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        padding: 5px !important;
    }
    
    label {
        color: #94a3b8 !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        margin-bottom: 8px !important;
    }

    /* بطاقات النتائج - تصميم مستقبلي */
    .metric-card {
        background: linear-gradient(145deg, #0f172a 0%, #020617 100%);
        border: 2px solid #1e293b;
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, transparent, #3b82f6, transparent);
    }
    
    .metric-card h3 {
        color: #3b82f6 !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        margin-bottom: 20px !important;
    }

    .metric-card h2 {
        color: #ffffff !important;
        font-size: 4rem !important;
        font-weight: 900 !important;
        margin: 0 !important;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.2);
    }

    /* إخفاء الهوامش الزائدة */
    .block-container {
        padding: 4rem 6rem !important;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. توليد بيانات وهمية (Synthetic Data Generation)
# -----------------------------------------------------------------------------
@st.cache_data
def generate_synthetic_data(n_samples=2000):
    np.random.seed(42)
    positions = ['GK', 'CB', 'LB', 'RB', 'CM', 'CAM', 'CDM', 'LW', 'RW', 'ST']
    feet = ['يمين', 'يسار']
    injury_levels = ['لا توجد', 'خفيفة', 'متوسطة', 'خطيرة']
    fame_levels = ['غير معروف', 'محلي', 'عالمي']
    contract_statuses = ['نعم', 'لا']
    match_statuses = ['أساسي', 'احتياطي', 'تدويري']
    
    data = {
        'age': np.random.randint(16, 40, n_samples),
        'height_cm': np.random.randint(160, 200, n_samples),
        'weight_kg': np.random.randint(60, 100, n_samples),
        'preferred_foot': np.random.choice(feet, n_samples),
        'position': np.random.choice(positions, n_samples),
        'pace': np.random.randint(40, 99, n_samples),
        'physic': np.random.randint(40, 99, n_samples),
        'shooting': np.random.randint(30, 99, n_samples),
        'passing': np.random.randint(40, 99, n_samples),
        'dribbling': np.random.randint(40, 99, n_samples),
        'controlling': np.random.randint(40, 99, n_samples),
        'discipline': np.random.randint(1, 11, n_samples),
        'is_injured': np.random.choice(['نعم', 'لا'], n_samples, p=[0.2, 0.8]),
        'injury_degree': np.random.choice(injury_levels, n_samples),
        'matches_played': np.random.randint(0, 50, n_samples),
        'goals': np.random.randint(0, 30, n_samples),
        'assists': np.random.randint(0, 20, n_samples),
        'participation_status': np.random.choice(match_statuses, n_samples),
        'fame_level': np.random.choice(fame_levels, n_samples, p=[0.5, 0.3, 0.2]),
        'has_contract': np.random.choice(contract_statuses, n_samples),
        'contract_years': np.random.randint(0, 6, n_samples),
        'league_strength': np.random.randint(1, 6, n_samples),
    }
    
    df = pd.DataFrame(data)
    fame_multiplier = df['fame_level'].map({'غير معروف': 1, 'محلي': 5, 'عالمي': 20})
    base_price = (df['pace'] * 1000 + df['shooting'] * 1500 + df['passing'] * 1200 + 
                  df['dribbling'] * 1300 + df['matches_played'] * 5000 + df['goals'] * 10000 + (40 - df['age']) * 20000)
    df['price'] = base_price * fame_multiplier * df['league_strength'] * 0.5
    df['price'] = df['price'] * df['injury_degree'].map({'لا توجد': 1, 'خفيفة': 0.9, 'متوسطة': 0.7, 'خطيرة': 0.4})
    df['price'] = df['price'] + np.random.normal(0, df['price']*0.1, n_samples)
    return df

# -----------------------------------------------------------------------------
# 2. بناء النموذج (Model Building)
# -----------------------------------------------------------------------------
@st.cache_resource
def build_model(df):
    X = df.drop('price', axis=1)
    y = df['price']
    numeric_features = ['age', 'height_cm', 'weight_kg', 'pace', 'physic', 'shooting', 'passing', 'dribbling', 'controlling', 'discipline', 'matches_played', 'goals', 'assists', 'contract_years', 'league_strength']
    categorical_features = ['preferred_foot', 'position', 'is_injured', 'injury_degree', 'participation_status', 'fame_level', 'has_contract']
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])
    
    model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    return model, r2_score(y_test, model.predict(X_test)), mean_absolute_error(y_test, model.predict(X_test)), X_train, y_train

# -----------------------------------------------------------------------------
# 3. واجهة المستخدم (UI Layout)
# -----------------------------------------------------------------------------

# تحميل البيانات والنموذج
with st.spinner('جاري التحليل...'):
    df_data = generate_synthetic_data(3000)
    model, r2_score_val, mae_val, X_train_ref, y_train_ref = build_model(df_data)

st.title("⚽ تحليل أسعار لاعبي كرة القدم")

# تم إزالة العنوان الفرعي ولوحة التحكم الجانبية بناءً على طلب المستخدم

# النموذج داخل Form لتنظيم المدخلات في أعمدة عمودية
with st.form("player_data_form"):
    
    # تقسيم المدخلات إلى 4 أعمدة رئيسية
    main_col1, main_col2, main_col3, main_col4 = st.columns(4)
    
    with main_col1:
        st.markdown("##### 👤 البيانات الشخصية")
        age = st.selectbox("العمر (سنة)", list(range(15, 46)), index=9)
        height = st.number_input("الطول (سم)", 150, 220, 180)
        weight = st.number_input("الوزن (كغ)", 50, 110, 75)
        position = st.selectbox("مركز اللعب", ['GK', 'CB', 'LB', 'RB', 'CM', 'CAM', 'CDM', 'LW', 'RW', 'ST'])
        foot = st.selectbox("القدم المفضلة", ['يمين', 'يسار'])

    with main_col2:
        st.markdown("##### ⚡ المهارات الفنية")
        pace = st.slider("السرعة", 0, 100, 70)
        shooting = st.slider("التسديد", 0, 100, 60)
        physic = st.slider("القوة البدنية", 0, 100, 75)
        passing = st.slider("التمرير", 0, 100, 65)
        dribbling = st.slider("المراوغة", 0, 100, 70)
        controlling = st.slider("التحكم بالكرة", 0, 100, 72)

    with main_col3:
        st.markdown("##### 📈 الأداء والانضباط")
        matches = st.number_input("عدد المباريات", 0, 100, 20)
        goals = st.number_input("عدد الأهداف", 0, 100, 5)
        assists = st.number_input("عدد الصناعات", 0, 100, 3)
        part_status = st.selectbox("حالة المشاركة", ['أساسي', 'احتياطي', 'تدويري'])
        discipline = st.slider("الانضباط (1-10)", 1, 10, 8)

    with main_col4:
        st.markdown("##### 🏥 الحالة والتعاقد")
        is_injured_val = st.radio("هل يعاني من إصابة؟", ['لا', 'نعم'], horizontal=True)
        injury_degree = st.selectbox("درجة الإصابة", ['لا توجد', 'خفيفة', 'متوسطة', 'خطيرة'])
        if is_injured_val == 'لا': injury_degree = 'لا توجد'
        
        fame = st.selectbox("مستوى الشهرة", ['غير معروف', 'محلي', 'عالمي'])
        league_str = st.slider("قوة الدوري (1-5)", 1, 5, 3)
        has_contract_val = st.radio("هل مرتبط بعقد؟", ['نعم', 'لا'], horizontal=True)
        contract_years = st.slider("سنوات العقد", 0, 10, 2) if has_contract_val == 'نعم' else 0

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🚀 تحليل وتوقع سعر اللاعب")

# -----------------------------------------------------------------------------
# 4. منطق التوقع وعرض النتائج
# -----------------------------------------------------------------------------
if submitted:
    input_df = pd.DataFrame([{
        'age': age, 'height_cm': height, 'weight_kg': weight, 'preferred_foot': foot, 'position': position,
        'pace': pace, 'physic': physic, 'shooting': shooting, 'passing': passing, 'dribbling': dribbling, 'controlling': controlling,
        'discipline': discipline, 'is_injured': is_injured_val, 'injury_degree': injury_degree,
        'matches_played': matches, 'goals': goals, 'assists': assists, 'participation_status': part_status,
        'fame_level': fame, 'has_contract': has_contract_val, 'contract_years': contract_years, 'league_strength': league_str
    }])
    
    predicted_price = max(0, model.predict(input_df)[0])
    
    if predicted_price < 1_000_000: level, color = "ضعيف", "gray"
    elif predicted_price < 10_000_000: level, color = "جيد", "blue"
    elif predicted_price < 50_000_000: level, color = "جيد جداً", "orange"
    else: level, color = "ممتاز", "green"

    st.markdown("---")
    res_c1, res_c2 = st.columns(2)
    with res_c1:
        st.markdown(f"<div class='metric-card'><h3>💰 السعر المتوقع</h3><h2 style='color:#00c04b !important;'>{predicted_price:,.0f} $</h2></div>", unsafe_allow_html=True)
    with res_c2:
        st.markdown(f"<div class='metric-card'><h3>⭐ تصنيف المستوى</h3><h2 style='color:{color} !important;'>{level}</h2></div>", unsafe_allow_html=True)

    # المقارنة الذكية
    same_pos_data = X_train_ref[X_train_ref['position'] == position].copy()
    same_pos_data['price'] = y_train_ref.loc[same_pos_data.index]
    avg_price = same_pos_data['price'].mean()
    diff = predicted_price - avg_price
    
    if diff < -avg_price * 0.2: verdict, v_color = "أقل من المتوسط (لقطة)", "#00c04b"
    elif diff > avg_price * 0.2: verdict, v_color = "أعلى من المتوسط (غالٍ)", "#ff4b4b"
    else: verdict, v_color = "سعر عادل", "#ffa500"

    st.markdown(f"<div class='metric-card'><h4>💡 مقارنة مع متوسط المركز ({position})</h4><p>متوسط السعر: {avg_price:,.0f} $</p><h3 style='color:{v_color} !important;'>{verdict}</h3></div>", unsafe_allow_html=True)

    # تصدير التقرير
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as workbook:
        input_df.assign(predicted_price=predicted_price, level=level, verdict=verdict).to_excel(workbook, sheet_name='Report', index=False)
    st.download_button("📄 تحميل التقرير (Excel)", output.getvalue(), f'report_{position}_{age}.xlsx', "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.caption("تم التطوير بواسطة: مساعد الذكاء الاصطناعي 🤖")
