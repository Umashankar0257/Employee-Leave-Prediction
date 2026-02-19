import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="Employee Leave Dashboard",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    /* Metric styling */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #eef0f2;
    }
    [data-testid="stMetricValue"] > div {
        color: #007bff !important;
    }
    [data-testid="stMetricLabel"] > div {
        color: #555555 !important;
    }
    
    /* Dropdown/Selectbox cursor */
    div[data-baseweb="select"] {
        cursor: pointer !important;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    
    /* Sidebar styling */
    .css-1r6slb0 {
        padding: 2rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache data and model loading
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("model.pkl")
        encoders = joblib.load("encoders.pkl")
        tfidf = joblib.load("tfidf.pkl")
        return model, encoders, tfidf
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("LEAVEHISTORY.csv")
        df['LVREQSTDT'] = pd.to_datetime(df['LVREQSTDT'], errors='coerce')
        df['LVFRDT'] = pd.to_datetime(df['LVFRDT'], errors='coerce')
        df['LVTODT'] = pd.to_datetime(df['LVTODT'], errors='coerce')
        df['applied_month'] = pd.to_numeric(df['applied_month'], errors='coerce').fillna(0).astype(int)
        return df
    except Exception as e:
        st.error(f"Error loading historical data: {e}")
        return pd.read_csv("CLEANED_DATASET.csv")

# Load artifacts
model, encoders, tfidf = load_artifacts()
df = load_data()

# --- SIDEBAR ---
st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("---")

# Dashboard Mode Selection
st.sidebar.subheader("Navigation")
dashboard_mode = st.sidebar.radio("Select View", ["Employee Analytics", "Absence Tracker"])
st.sidebar.markdown("---")

# Global Date Filter (Consolidated)
st.sidebar.subheader("Filter by Date Range")
global_min = df['LVREQSTDT'].min().date() if not df['LVREQSTDT'].isnull().all() else datetime.now().date()
# Allow selection up to the end of 2027 for future absence tracking
global_max = datetime(2027, 12, 31).date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(global_min, global_max),
    min_value=global_min,
    max_value=global_max,
    help="Filter analytics or tracking by request/leave dates."
)

start_date, end_date = None, None
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range

if dashboard_mode == "Employee Analytics":
    # Employee Selection
    all_employees = sorted(df['EMPLOYEECODE'].unique())
    selected_employee = st.sidebar.selectbox("Select Employee ID", all_employees)
    
    emp_all_data = df[df['EMPLOYEECODE'] == selected_employee]
    
    # Filtering Logic for Employee Analytics
    if start_date and end_date:
        emp_df = emp_all_data[
            (emp_all_data['LVREQSTDT'].dt.date >= start_date) & 
            (emp_all_data['LVREQSTDT'].dt.date <= end_date)
        ].copy()
        filter_desc = f"Results for: **{start_date}** to **{end_date}**"
    else:
        emp_df = emp_all_data.copy()
        filter_desc = "Showing **Overall History** for this employee."

    # --- MAIN DASHBOARD (Indented!) ---
    st.title(f"👤 Employee Leave Analysis: {selected_employee}")
    st.caption(filter_desc)
    
    if emp_df.empty:
        st.warning("No historical data found for this employee in the selected criteria.")
    else:
        # Top Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_requests = len(emp_df)
        approved_df = emp_df[emp_df['approval_status'] == 1]
        total_approved = len(approved_df)
        approval_rate = (total_approved / total_requests) * 100 if total_requests > 0 else 0
        
        # Most frequent leave code
        if not emp_df['LEAVECODE'].empty:
            most_freq_leave = emp_df['LEAVECODE'].mode()[0]
        else:
            most_freq_leave = "N/A"
            
        with col1:
            st.metric("Total Requests", total_requests)
        with col2:
            st.metric("Total Approved", total_approved)
        with col3:
            st.metric("Approval Rate", f"{approval_rate:.1f}%")
        with col4:
            st.metric("Most Frequent Type", most_freq_leave)

        st.markdown("---")
        
        # --- LEAVE BALANCES TABLE (Always Latest Data) ---
        st.subheader("💳 Current (Latest) Leave Balances")
        if not emp_all_data.empty:
            last_rec = emp_all_data.sort_values('LVREQSTDT').iloc[-1]
            balance_data = {
                "Item": ["Current EL Balance", "Current HPL Balance", "Total Requests (All Time)", "Total Approved (All Time)"],
                "Value": [
                    f"{last_rec['ELBALANCE']} days", 
                    f"{last_rec['HPLBALANCE']} days", 
                    len(emp_all_data),
                    len(emp_all_data[emp_all_data['approval_status'] == 1])
                ]
            }
            st.table(pd.DataFrame(balance_data))
        else:
            st.info("No balance history found for this employee.")

        st.markdown("---")
        
        # --- DETAILED LEAVE HISTORY TABLE ---
        st.subheader("📋 Detailed Leave History (Selected Period)")
        if not emp_df.empty:
            detail_df = emp_df.copy()
            # Map approval status for readability
            detail_df['Status'] = detail_df['approval_status'].map({1: '✅ Approved', 0: '❌ Rejected'})
            # Rename columns for display
            display_columns = {
                'LEAVECODE': 'Leave Type',
                'LVFRDT': 'From Date',
                'LVTODT': 'To Date',
                'leave_days': 'Days',
                'Status': 'Status',
                'LVREASON': 'Reason'
            }
            # Select and format columns
            final_detail_df = detail_df[list(display_columns.keys())].rename(columns=display_columns)
            # Format dates to string for better display in table
            final_detail_df['From Date'] = final_detail_df['From Date'].dt.strftime('%Y-%m-%d')
            final_detail_df['To Date'] = final_detail_df['To Date'].dt.strftime('%Y-%m-%d')
            
            st.dataframe(final_detail_df, use_container_width=True, hide_index=True)
        else:
            st.info("No detailed records found for the selected dates.")

        st.markdown("---")

        # Visualizations Row 1: Distribution and Approval Breakdown
        vcol1, vcol2 = st.columns(2)
        
        with vcol1:
            st.subheader("📊 Leave Type Distribution")
            fig_pie = px.pie(emp_df, names='LEAVECODE', hole=0.5, 
                             color_discrete_sequence=px.colors.qualitative.Safe)
            fig_pie.update_layout(showlegend=True, margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with vcol2:
            st.subheader("✅ Approval Status by Leave Type")
            status_map = {1: 'Approved', 0: 'Rejected'}
            breakdown_df = emp_df.copy()
            breakdown_df['Status'] = breakdown_df['approval_status'].map(status_map)
            
            fig_breakdown = px.bar(breakdown_df, x='LEAVECODE', color='Status',
                                   title="Detailed Approval Count",
                                   barmode='stack',
                                   color_discrete_map={'Approved': '#2ecc71', 'Rejected': '#e74c3c'},
                                   category_orders={"Status": ["Approved", "Rejected"]})
            fig_breakdown.update_layout(xaxis_title="Leave Type", yaxis_title="Number of Requests")
            st.plotly_chart(fig_breakdown, use_container_width=True)

        # Visualizations Row 2: Trends
        st.markdown("---")
        st.subheader("📅 Monthly Leave Volume Trends")
        
        month_names = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        
        monthly_pattern = emp_df.groupby('applied_month').size().reset_index(name='Requests')
        if not monthly_pattern.empty:
            monthly_pattern['applied_month'] = monthly_pattern['applied_month'].astype(int)
            monthly_pattern['Month'] = monthly_pattern['applied_month'].map(month_names)
            monthly_pattern = monthly_pattern.sort_values('applied_month')
            
            fig_line = px.bar(monthly_pattern, x='Month', y='Requests',
                               text='Requests',
                               title="Requests per Month",
                               color_discrete_sequence=['#3498db'])
            
            fig_line.add_trace(go.Scatter(x=monthly_pattern['Month'], y=monthly_pattern['Requests'],
                                          mode='lines+markers', name='Trend',
                                          line=dict(color='#2c3e50', width=2)))
            
            fig_line.update_traces(textposition='outside', selector=dict(type='bar'))
            fig_line.update_layout(xaxis_title="Month", yaxis_title="Total Requests",
                                   showlegend=False)
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No monthly data available for the selected filters.")

    st.markdown("---")

    # --- PREDICTION MODULE ---
    st.header("Apply for Leave Simulator")
    st.markdown("Simulate a new leave request to predict approval probability.")

    pcol1, pcol2 = st.columns(2)

    with pcol1:
        reason = st.text_area("Reason for Leave", placeholder="e.g., Family vacation, Sick leave...")
        leave_code = st.selectbox("Leave Type", options=encoders['LEAVECODE'].classes_)
        leave_days = st.number_input("Number of Days", min_value=1, max_value=30, value=1)

    with pcol2:
        target_date = st.date_input("Applied Date", datetime.now())
        day_of_week = float(target_date.weekday())
        month = float(target_date.month)
        
        last_record = emp_df.iloc[-1] if not emp_df.empty else None
        el_balance = st.number_input("Current EL Balance", value=float(last_record['ELBALANCE']) if last_record is not None else 10.0)
        hpl_balance = st.number_input("Current HPL Balance", value=float(last_record['HPLBALANCE']) if last_record is not None else 5.0)
        past_leaves = len(emp_df)

    if st.button("Predict Approval Probability"):
        if not reason:
            st.error("Please provide a reason for the leave.")
        else:
            try:
                enc_leave = encoders['LEAVECODE'].transform([str(leave_code)])[0]
                enc_day = encoders['applied_day_of_week'].transform([str(day_of_week)])[0]
                enc_month = encoders['applied_month'].transform([str(month)])[0]
                
                numeric_features = [enc_leave, enc_day, enc_month, leave_days, el_balance, hpl_balance, past_leaves]
                reason_tfidf = tfidf.transform([reason]).toarray()
                X_input = np.hstack(([numeric_features], reason_tfidf))
                
                # Policy Engine
                policy_rejected = False
                policy_warning = False
                rejection_reason = ""
                warning_msg = ""
                
                requests_this_month = emp_all_data[emp_all_data['applied_month'] == month]
                if len(requests_this_month) >= 3:
                    policy_rejected = True
                    rejection_reason = f"Frequency Limit: You have already made {len(requests_this_month)} requests in {month_names.get(int(month))}. Maximum allowed is 3."

                same_type_this_month = emp_all_data[
                    (emp_all_data['LEAVECODE'] == leave_code) & 
                    (emp_all_data['applied_month'] == month) &
                    (emp_all_data['approval_status'] == 1)
                ]
                if len(same_type_this_month) >= 1:
                    policy_rejected = True
                    rejection_reason = f"Strict Policy: You have already taken {leave_code} in {month_names.get(int(month))}. Only one {leave_code} is allowed per month."

                if leave_code == "EL":
                    if leave_days > 15:
                        policy_rejected = True
                        rejection_reason = "EL Policy: Earned Leave cannot exceed 15 consecutive days in a single application."
                    if el_balance < leave_days:
                        policy_rejected = True
                        rejection_reason = f"Balance Error: Your EL balance ({el_balance}) is less than the requested days ({leave_days})."

                if leave_code == "CMC":
                    if leave_days > 7:
                        policy_warning = True
                        warning_msg = "Caution: CMC requests exceeding 7 days require a Senior Medical Certificate for approval."

                if leave_code == "HPL":
                    if hpl_balance < leave_days:
                        policy_rejected = True
                        rejection_reason = f"Balance Error: Your HPL balance ({hpl_balance}) is less than the requested days ({leave_days})."

                if policy_rejected:
                    prob = 0.0
                else:
                    prob = model.predict_proba(X_input)[0][1]
                    if policy_warning:
                        prob = min(prob, 0.4)

                st.markdown("---")
                if policy_rejected:
                    st.error(f"### ❌ Request Rejected: 100% Certainty")
                    st.write(f"**Reason:** {rejection_reason}")
                elif policy_warning:
                    st.warning(f"### ⚠️ Careful: {prob*100:.1f}% Likelihood")
                    st.write(f"**Notice:** {warning_msg}")
                elif prob > 0.8:
                    st.success(f"### High likelihood of approval: {prob*100:.1f}%")
                elif prob > 0.5:
                    st.info(f"### Moderate likelihood of approval: {prob*100:.1f}%")
                else:
                    st.warning(f"### Low likelihood of approval: {prob*100:.1f}%")
                    
                st.write("**Risk Analysis:**")
                if policy_rejected:
                    st.write(f"- 🚩 **STRICT REJECTION**: {rejection_reason}")
                if policy_warning:
                    st.write(f"- 🔔 **POLICY NOTICE**: {warning_msg}")
                if el_balance < leave_days and leave_code != "EL":
                    st.write("- 🚩 Low EL Balance.")
                if past_leaves > 15:
                    st.write("- 🚩 Frequent past leave requests detected.")
                if len(reason.split()) < 3:
                    st.write("- 🚩 Vague reason provided.")
                    
            except Exception as e:
                st.error(f"Prediction error: {e}")


# --- ABSENCE TRACKER MODE ---
else:
    st.title("🚨 Absence Tracker")
    st.markdown("Track which employees are absent on a specific day or date range.")
    
    # Sidebar Controls for Absence Tracker
    st.sidebar.subheader("Tracker Settings")
    track_type = st.sidebar.radio("Tracking Type", ["Single Day", "Date Range"])
    
    if track_type == "Single Day":
        # We use the start_date from the global picker as the single selected date
        selected_date = st.sidebar.date_input("Select Date", start_date if start_date else datetime.now().date())
        # Filter for approved leaves that cover the selected date
        mask = (df['approval_status'] == 1) & \
               (df['LVFRDT'].dt.date <= selected_date) & \
               (df['LVTODT'].dt.date >= selected_date)
        
        absentees = df[mask].drop_duplicates('EMPLOYEECODE')
        
        st.subheader(f"Absentees on {selected_date}")
        st.metric("Total Absent", len(absentees))
        
        if not absentees.empty:
            st.markdown("### 📋 Absent Employee Codes")
            cols = st.columns(5)
            for i, emp_code in enumerate(absentees['EMPLOYEECODE']):
                cols[i % 5].info(f"**{emp_code}**")
            
            st.markdown("---")
            st.subheader("Detailed Leave Information")
            st.dataframe(absentees[['EMPLOYEECODE', 'LEAVECODE', 'LVREASON', 'LVFRDT', 'LVTODT']])
        else:
            st.info(f"Everyone is present on {selected_date}!")

    else:
        # Use the global sidebar dates
        if start_date and end_date:
            if start_date > end_date:
                st.error("Error: 'From Date' must be before 'To Date'.")
            else:
                mask = (df['approval_status'] == 1) & \
                       (df['LVFRDT'].dt.date <= end_date) & \
                       (df['LVTODT'].dt.date >= start_date)
                
                overlap_df = df[mask].copy()
                
                if not overlap_df.empty:
                    def calc_days_in_range(row):
                        actual_start = max(row['LVFRDT'].date(), start_date)
                        actual_end = min(row['LVTODT'].date(), end_date)
                        delta = (actual_end - actual_start).days + 1
                        return max(0, delta)
                    
                    overlap_df['days_absent_in_range'] = overlap_df.apply(calc_days_in_range, axis=1)
                    
                    summary = overlap_df.groupby('EMPLOYEECODE').agg({
                        'days_absent_in_range': 'sum',
                        'LEAVECODE': lambda x: ', '.join(x.unique())
                    }).reset_index()
                    
                    summary.columns = ['Employee Code', 'Total Days Absent', 'Leave Types']
                    
                    st.subheader(f"Absence Summary: {start_date} to {end_date}")
                    st.metric("Unique Employees Absent", len(summary))
                    
                    st.markdown("### 📊 Absence Data")
                    st.table(summary.sort_values(by='Total Days Absent', ascending=False))
                    
                    with st.expander("View Raw Overlap Details"):
                        st.dataframe(overlap_df[['EMPLOYEECODE', 'LEAVECODE', 'LVFRDT', 'LVTODT', 'days_absent_in_range']])
                else:
                    st.info(f"No absentees found between {start_date} and {end_date}.")
        else:
            st.sidebar.warning("Please select a valid date range in the sidebar.")

st.markdown("---")
st.caption("Developed for Employee Leave Prediction Analysis • 2026")
