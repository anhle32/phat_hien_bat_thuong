import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import io

# -----------------------------------------------------------------------------
# STEP 1: CONFIGURATION & FIRST LUNCH
# -----------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Hệ Thống Phát Hiện Giao Dịch Bất Thường",
    page_icon="🛡️"
)

# Khởi tạo session state nếu chưa có để tránh mất trạng thái khi chuyển tab
if "model" not in st.session_state:
    st.session_state.model = None
if "preprocessor" not in st.session_state:
    st.session_state.preprocessor = None
if "df_scored" not in st.session_state:
    st.session_state.df_scored = None
if "trained_features" not in st.session_state:
    st.session_state.trained_features = None

# -----------------------------------------------------------------------------
# STEP 2: CACHED DATA LOADING & PREPROCESSING UTILS
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Đang xử lý cấu trúc dữ liệu...")
def load_and_preprocess_data(file_bytes):
    """
    Nạp dữ liệu từ bytes, ép kiểu thời gian và tạo các biến phái sinh 
    giống như quy trình trong notebook.
    """
    df = pd.read_csv(io.BytesIO(file_bytes))
    
    # Ép kiểu ngày tháng
    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], format='%d/%m/%Y %H:%M', errors='coerce')
        # Tạo biến phái sinh thời gian phục vụ phân tích
        df['hour'] = df['transaction_date'].dt.hour
        df['day_of_week'] = df['transaction_date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
    return df

# -----------------------------------------------------------------------------
# STEP 3: SIDEBAR - CONFIGURATION ZONE
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # 1. Tải file dữ liệu mẫu
    uploaded_file = st.file_uploader(
        "Tải lên tệp dữ liệu giao dịch (.csv)", 
        type=["csv"],
        help="Chọn tệp CSV chứa lịch sử giao dịch quý 1 để phân tích và huấn luyện mô hình."
    )
    
    st.divider()
    st.subheader("Tham số mô hình AI")
    st.caption("Thuật toán mặc định: **Isolation Forest**")
    
    # 2. Cấu hình siêu tham số từ notebook
    contamination = st.slider(
        "Tỷ lệ nhiễm bẩn (Contamination)",
        min_value=0.005,
        max_value=0.150,
        value=0.015,
        step=0.005,
        help="Tỷ lệ ước tính của các điểm dữ liệu bất thường trong tập dữ liệu."
    )
    
    n_estimators = st.slider(
        "Số lượng cây (n_estimators)",
        min_value=50,
        max_value=300,
        value=100,
        step=50,
        help="Số lượng cây quyết định cô lập trong rừng."
    )
    
    max_features = st.slider(
        "Tỷ lệ biến số lấy mẫu (max_features)",
        min_value=0.5,
        max_value=1.0,
        value=1.0,
        step=0.1,
        help="Tỷ lệ số lượng biến được lấy mẫu ngẫu nhiên từ tập dữ liệu để huấn luyện mỗi cây."
    )
    
    # Gom tham số nâng cao vào expander
    with st.expander("⚙️ Cấu hình nâng cao"):
        random_state = st.number_input(
            "Mã ngẫu nhiên (random_state)",
            min_value=0,
            max_value=9999,
            value=42,
            step=1,
            help="Giá trị cố định để tái lập kết quả huấn luyện mô hình."
        )
        
    st.divider()
    
    # 3. Nút hành động duy nhất để huấn luyện
    trigger_train = st.button(
        "🚀 Huấn Luyện & Phát Hiện", 
        type="primary", 
        use_container_width=True,
        help="Bấm để chạy quy trình tiền xử lý dữ liệu và huấn luyện mô hình Isolation Forest."
    )

# -----------------------------------------------------------------------------
# STEP 4: HEADER & DATA VALIDATION
# -----------------------------------------------------------------------------
st.title("🛡️ Hệ Thống Phát Hiện Giao Dịch Bất Thường (Anomaly Detection)")
st.caption("Ứng dụng hỗ trợ phòng chống gian lận, phát hiện tự động các hành vi giao dịch tài chính bất thường dựa trên mô hình học máy Isolation Forest.")

if uploaded_file is None:
    st.info("💡 Vui lòng tải lên tệp dữ liệu giao dịch ở thanh Sidebar bên trái để bắt đầu.")
    st.stop()

# Đọc dữ liệu qua cache hàm dùng chung
try:
    df_raw = load_and_preprocess_data(uploaded_file.getvalue())
    st.caption(f"📁 Đang dùng tệp dữ liệu: `{uploaded_file.name}` | Tổng số dòng dữ liệu: **{df_raw.shape[0]:,}**")
except Exception as e:
    st.error(f"Xảy ra lỗi khi đọc định dạng tệp CSV: {e}")
    st.stop()

st.divider()

# -----------------------------------------------------------------------------
# STEP 5: MODEL TRAINING CORE ENGINE (EXECUTE ON BUTTON CLICK)
# -----------------------------------------------------------------------------
# Định nghĩa các đặc trưng đưa vào mô hình dựa trên tệp mẫu và notebook
features_numeric = ['amount', 'hour', 'day_of_week']
features_categorical = ['transaction_type', 'channel', 'location']
all_model_features = features_numeric + features_categorical

if trigger_train:
    with st.spinner("Đang xử lý Pipeline huấn luyện mô hình Isolation Forest..."):
        try:
            # Kiểm tra xem dữ liệu thô có đủ cột đặc trưng không
            missing_cols = [col for col in all_model_features if col not in df_raw.columns]
            if missing_cols:
                st.error(f"Dữ liệu tải lên thiếu các cột bắt buộc cho mô hình: {missing_cols}")
                st.stop()
                
            # Tạo bản sao để xử lý huấn luyện
            df_train = df_raw.copy()
            
            # Xử lý các giá trị khuyết thiếu cơ bản nếu có
            for col in features_numeric:
                df_train[col] = df_train[col].fillna(df_train[col].median())
            for col in features_categorical:
                df_train[col] = df_train[col].fillna(df_train[col].mode()[0])
            
            # Thiết lập bộ tiền xử lý chuẩn hóa dữ liệu
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), features_numeric),
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_categorical)
                ]
            )
            
            # Biến đổi dữ liệu X
            X_encoded = preprocessor.fit_transform(df_train[all_model_features])
            
            # Khởi tạo mô hình học máy theo tham số người dùng chọn
            model = IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                max_features=max_features,
                random_state=random_state,
                n_jobs=-1
            )
            
            # Huấn luyện mô hình
            model.fit(X_encoded)
            
            # Chấm điểm và gán nhãn bất thường
            # isolation forest: -1 là bất thường, 1 là bình thường
            preds = model.predict(X_encoded)
            scores = model.decision_function(X_encoded) # Điểm số càng âm càng bất thường
            
            df_train['anomaly_score'] = scores
            df_train['is_anomaly'] = np.where(preds == -1, True, False)
            
            # Lưu trữ toàn bộ kết quả vào session_state để tái sử dụng xuyên suốt các Tab
            st.session_state.model = model
            st.session_state.preprocessor = preprocessor
            st.session_state.df_scored = df_train
            st.session_state.trained_features = all_model_features
            
            st.success("✅ Huấn luyện mô hình thành công! Hãy chuyển qua các tab bên dưới để xem kết quả chi tiết.")
            
        except Exception as e:
            st.error(f"Lỗi trong quá trình huấn luyện: {str(e)}")

# -----------------------------------------------------------------------------
# STEP 6: APP TABS SECTION
# -----------------------------------------------------------------------------
tab_summary, tab_viz, tab_results, tab_inference = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa biến số", 
    "🎯 Kết quả phát hiện bất thường", 
    "🔮 Trực tuyến & Dự báo lô"
])

# -----------------------------------------------------------------------------
# TAB 3: TỔNG QUAN DỮ LIỆU
# -----------------------------------------------------------------------------
with tab_summary:
    st.subheader("Tổng quan tập dữ liệu giao dịch đầu vào")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Số hàng dữ liệu", f"{df_raw.shape[0]:,}")
    col2.metric("Số cột dữ liệu", f"{df_raw.shape[1]}")
    col3.metric("Dung lượng tệp", f"{uploaded_file.size / (1024*1024):.2f} MB")
    
    st.write("##### 🕵️ Xem trước 5 dòng dữ liệu đầu tiên:")
    st.dataframe(df_raw.head(5), height=230, use_container_width=True)
    
    st.write("##### 📉 Thống kê mô tả các biến số đưa vào mô hình:")
    # Chỉ mô tả các biến cốt lõi cho mô hình
    available_numeric = [c for c in features_numeric if c in df_raw.columns]
    st.dataframe(df_raw[available_numeric].describe().T, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: TRỰC QUAN HÓA BIẾN SỐ
# -----------------------------------------------------------------------------
with tab_viz:
    st.subheader("Phân tích trực quan các biến số chính")
    
    # Lưới hiển thị biểu đồ 2x2 cho các biến quan trọng
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    
    # 1. Biểu đồ phân bố số tiền giao dịch (amount)
    with row1_col1:
        fig_amount = px.histogram(
            df_raw, x='amount', nbins=50, 
            title="Phân phối số tiền giao dịch (Amount Distribution)",
            labels={'amount': 'Số tiền (VND)'},
            color_discrete_sequence=['#00CC96']
        )
        fig_amount.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_amount, use_container_width=True)
        
    # 2. Biểu đồ phân phối theo khung giờ trong ngày (hour)
    with row1_col2:
        if 'hour' in df_raw.columns:
            fig_hour = px.histogram(
                df_raw, x='hour', nbins=24,
                title="Số lượng giao dịch phân bố theo khung giờ trong ngày",
                labels={'hour': 'Giờ (0 - 23)'},
                color_discrete_sequence=['#636EFA']
            )
            fig_hour.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_hour, use_container_width=True)
            
    # 3. Phân phối loại hình giao dịch (transaction_type)
    with row2_col1:
        if 'transaction_type' in df_raw.columns:
            df_type = df_raw['transaction_type'].value_counts().reset_index()
            fig_type = px.bar(
                df_type, x='transaction_type', y='count',
                title="Tần suất theo loại giao dịch",
                labels={'transaction_type': 'Loại giao dịch', 'count': 'Số lượng'},
                color='transaction_type', color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_type.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
            st.plotly_chart(fig_type, use_container_width=True)

    # 4. Phân phối theo kênh giao dịch (channel)
    with row2_col2:
        if 'channel' in df_raw.columns:
            df_channel = df_raw['channel'].value_counts().reset_index()
            fig_channel = px.bar(
                df_channel, x='channel', y='count',
                title="Tần suất theo kênh giao dịch (Channel)",
                labels={'channel': 'Kênh giao dịch', 'count': 'Số lượng'},
                color='channel', color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_channel.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
            st.plotly_chart(fig_channel, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5: KẾT QUẢ HUÂN LUYỆN & KIỂM ĐỊNH MÔ HÌNH
# -----------------------------------------------------------------------------
with tab_results:
    st.subheader("Kết quả cấu trúc rủi ro & Phát hiện bất thường")
    
    # Điều phối kiểm tra trạng thái session_state
    if st.session_state.df_scored is None:
        st.info("📢 Hãy nhấn nút '🚀 Huấn Luyện & Phát Hiện' tại thanh Sidebar bên trái để chạy phân tích mô hình AI.")
    else:
        df_res = st.session_state.df_scored
        
        # Tính toán các chỉ số thống kê bất thường dựa trên mô hình đã fit
        total_records = len(df_res)
        num_anomalies = df_res['is_anomaly'].sum()
        pct_anomalies = (num_anomalies / total_records) * 100
        
        # Trình bày chỉ tiêu vô hướng ra giao diện web
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Tổng số giao dịch phân tích", f"{total_records:,}")
        res_col2.metric("Số giao dịch bất thường phát hiện", f"{num_anomalies:,}", delta=f"-{pct_anomalies:.2f}%", delta_color="inverse")
        res_col3.metric("Ngưỡng điểm bất thường tối đa", f"{df_res['anomaly_score'].min():.4f}")
        
        st.divider()
        
        res_viz1, res_viz2 = st.columns(2)
        
        # Biểu đồ mật độ phân phối điểm Anomaly Score
        with res_viz1:
            fig_score_dist = px.histogram(
                df_res, x='anomaly_score', color='is_anomaly',
                title="Phân phối điểm số bất thường (Màu đỏ đại diện cho mẫu bị cô lập)",
                labels={'anomaly_score': 'Mức độ tin cậy/Điểm số cô lập', 'is_anomaly': 'Bất thường'},
                color_discrete_map={True: '#EF553B', False: '#636EFA'},
                barmode='overlay'
            )
            st.plotly_chart(fig_score_dist, use_container_width=True)
            
        # Biểu đồ tán xạ trực quan hóa mối quan hệ giữa Số tiền và Giờ giao dịch đánh dấu điểm bất thường
        with res_viz2:
            fig_scatter = px.scatter(
                df_res, x='hour', y='amount', color='is_anomaly',
                title="Bản đồ định vị giao dịch: Số tiền vs Giờ giao dịch",
                labels={'hour': 'Khung giờ', 'amount': 'Số tiền (VND)', 'is_anomaly': 'Trạng thái bất thường'},
                color_discrete_map={True: '#EF553B', False: '#AB63FA'},
                opacity=0.6
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        st.write("##### 🚨 Danh sách chi tiết các giao dịch có độ rủi ro cao (Bất thường):")
        df_only_anomalies = df_res[df_res['is_anomaly'] == True].sort_values(by='anomaly_score')
        st.dataframe(df_only_anomalies, height=300, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 6: SỬ DỤNG MÔ HÌNH (DỰ BÁO TRỰC TUYẾN & THEO LÔ)
# -----------------------------------------------------------------------------
with tab_inference:
    st.subheader("Ứng dụng mô hình vào kiểm định giao dịch mới")
    
    if st.session_state.model is None or st.session_state.preprocessor is None:
        st.info("📢 Hãy nhấn nút '🚀 Huấn Luyện & Phát Hiện' tại thanh Sidebar bên trái để sẵn sàng sử dụng chức năng dự báo.")
    else:
        # Lấy mô hình và bộ tiền xử lý từ session state ra dùng
        model = st.session_state.model
        preprocessor = st.session_state.preprocessor
        trained_features = st.session_state.trained_features
        
        mode = st.radio(
            "Chọn phương thức kiểm định rủi ro:",
            options=["Chế độ 1: Kiểm định giao dịch trực tuyến (Nhập tay)", "Chế độ 2: Kiểm định danh sách hàng loạt (Tải file mới)"],
            horizontal=True
        )
        
        # ---------------------------------------------------------
        # CHẾ ĐỘ 1: NHẬP TRỰC TIẾP QUA FORM
        # ---------------------------------------------------------
        if mode == "Chế độ 1: Kiểm định giao dịch trực tuyến (Nhập tay)":
            st.write("#### 📝 Nhập thông tin chi tiết giao dịch mới:")
            
            with st.form("online_inference_form"):
                col_inf1, col_inf2, col_inf3 = st.columns(3)
                
                with col_inf1:
                    input_amount = st.number_input(
                        "Số tiền giao dịch (VND):", 
                        min_value=0, 
                        value=int(df_raw['amount'].median()),
                        step=50000
                    )
                    input_hour = st.slider(
                        "Khung giờ giao dịch (0-23):", 
                        min_value=0, max_value=23, 
                        value=12
                    )
                    input_day = st.slider(
                        "Ngày trong tuần (0: Thứ 2 -> 6: Chủ nhật):", 
                        min_value=0, max_value=6, 
                        value=2
                    )
                    
                with col_inf2:
                    tx_types = df_raw['transaction_type'].dropna().unique().tolist() if 'transaction_type' in df_raw.columns else ['TRANSFER']
                    input_type = st.selectbox("Loại hình giao dịch:", options=tx_types)
                    
                    channels = df_raw['channel'].dropna().unique().tolist() if 'channel' in df_raw.columns else ['INTERNET']
                    input_channel = st.selectbox("Kênh thực hiện giao dịch:", options=channels)
                    
                with col_inf3:
                    locations = df_raw['location'].dropna().unique().tolist() if 'location' in df_raw.columns else ['CN TP.HCM']
                    input_location = st.selectbox("Chi nhánh / Vị trí:", options=locations)
                
                submit_predict = st.form_submit_button("🛡️ Kiểm tra mức độ rủi ro")
                
            if submit_predict:
                # Xây dựng DataFrame dòng đơn từ thông tin nhập vào
                df_single = pd.DataFrame([{
                    'amount': input_amount,
                    'hour': input_hour,
                    'day_of_week': input_day,
                    'transaction_type': input_type,
                    'channel': input_channel,
                    'location': input_location
                }])
                
                # Áp dụng chính xác bộ mã hóa tiền xử lý lúc train
                X_single_encoded = preprocessor.transform(df_single[trained_features])
                
                # Thực hiện dự đoán từ mô hình đã lưu
                pred_single = model.predict(X_single_encoded)[0]
                score_single = model.decision_function(X_single_encoded)[0]
                
                st.write("#### Kết quả phân tích rủi ro hệ thống:")
                if pred_single == -1:
                    st.error(f"🚨 CẢNH BÁO: Giao dịch này có hành vi **BẤT THƯỜNG / RỦI RO CAO**! (Điểm số cô lập: {score_single:.4f})")
                else:
                    st.success(f"🟢 AN TOÀN: Giao dịch nằm trong ngưỡng hành vi **BÌNH THƯỜNG**. (Điểm số cô lập: {score_single:.4f})")
                    
        # ---------------------------------------------------------
        # CHẾ ĐỘ 2: TẢI FILE MỚI KIỂM TRA HÀNG LOẠT
        # ---------------------------------------------------------
        else:
            st.write("#### 📂 Tải lên danh sách các giao dịch mới cần quét rủi ro:")
            new_file = st.file_uploader(
                "Yêu cầu định dạng tệp đồng cấu trúc với tập huấn luyện", 
                type=["csv"], 
                key="inference_bulk_file"
            )
            
            if new_file is not None:
                try:
                    df_new = load_and_preprocess_data(new_file.getvalue())
                    
                    # Kiểm tra tính khớp của các đặc trưng đầu vào
                    missing_fields = [f for f in trained_features if f not in df_new.columns]
                    if missing_fields:
                        st.error(f"Lỗi cấu trúc tệp dữ liệu! File tải lên thiếu các cột đầu vào bắt buộc sau: {missing_fields}")
                    else:
                        # Điền giá trị khuyết nếu có
                        for col in features_numeric:
                            df_new[col] = df_new[col].fillna(df_raw[col].median())
                        for col in features_categorical:
                            df_new[col] = df_new[col].fillna(df_raw[col].mode()[0])
                            
                        # Chạy quy trình tiền xử lý và chấm điểm mô hình
                        X_new_encoded = preprocessor.transform(df_new[trained_features])
                        preds_new = model.predict(X_new_encoded)
                        scores_new = model.decision_function(X_new_encoded)
                        
                        df_new['anomaly_score'] = scores_new
                        df_new['is_anomaly'] = np.where(preds_new == -1, True, False)
                        
                        # Thống kê nhanh số ca phát hiện rủi ro mới
                        new_anoms = df_new['is_anomaly'].sum()
                        st.warning(f"🕵️ Kết quả quét: Đã phát hiện thấy **{new_anoms}** giao dịch bất thường trong số **{len(df_new)}** bản ghi được tải lên.")
                        
                        # Cho phép xem kết quả trực tiếp và tải xuống file đính kèm nhãn
                        st.dataframe(df_new, height=250, use_container_width=True)
                        
                        # Xuất file kết quả dạng CSV hỗ trợ tiếng Việt (utf-8-sig)
                        csv_buffer = io.StringIO()
                        df_new.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                        
                        st.download_button(
                            label="📥 Tải xuống bảng kết quả gắn nhãn rủi ro (.csv)",
                            data=csv_buffer.getvalue(),
                            file_name="ket_qua_phat_hien_bat_thuong_giao_dich.csv",
                            mime="text/csv"
                        )
                except Exception as ex:
                    st.error(f"Xảy ra lỗi trong tiến trình dự báo hàng loạt: {ex}")
