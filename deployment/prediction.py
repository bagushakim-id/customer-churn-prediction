from keras import backend as K
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import calendar


# Custom Metrics

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Load All Files


with open('final_pipeline.pkl', 'rb') as file_1:
    model_pipeline = pickle.load(file_1)

# model_ann = load_model('churn_model.h5')
model_ann = load_model("churn_model.h5", custom_objects={"f1_m": f1_m})


def run():

    with st.form(key='Churn_Form'):
        user_id = st.text_input('User_ID', value='')
        age = st.number_input('Age', min_value=16, max_value=60,
                              value=25, step=2, help='Usia Pemain')
        gender = st.radio('Gender', ('Male', 'Female'), index=1)
        region_category = st.selectbox(
            'Region', ('Town', 'City', 'Village'), index=1)
        membership_category = st.selectbox('Membership', ('No Membership', 'Basic Membership', 'Silver Membership',
                                                          'Gold Membership', 'Premium Membership', 'Platinum Membership'),
                                           index=1)
        st.markdown('---')

        joining_date = st.date_input('Joining Date', value=None, min_value=None, max_value=None, key=None, help=None,
                                     on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        joined_through_referral = st.selectbox(
            'Join with Referral?', ('Yes', 'No', 'Village'), index=1)
        preferred_offer_types = st.selectbox('Preffered Offer Types', ('Gift Vouchers/Coupons', 'Credit/Debit Card Offers',
                                                                       'Without Offers'), index=1)
        medium_of_operation = st.selectbox(
            'Device', ('Desktop', 'Smartphone', 'Both'), index=1)
        internet_option = st.selectbox(
            'Internet', ('Wi-Fi', 'Wi-Fi', 'Fiber_Optic'), index=1)
        last_visit_time = st.time_input('Last Visit Time')
        days_since_last_login = st.slider('Last Login in days', 0, 31, 10)
        avg_time_spent = st.slider('Average of Time Spent', 0, 3600, 10)
        avg_transaction_value = st.slider(
            'Average of Transaction', 0, 100000, 10)
        avg_frequency_login_days = st.slider(
            'Average of Login Frequency on Days', 0, 1000, 10)
        points_in_wallet = st.slider('Points in Wallet', 0, 1000, 10)

        used_special_discount = st.radio(
            'Use Special Discount?', ('Yes', 'No'), index=1)
        offer_application_preference = st.radio(
            'offer application Preference?', ('Yes', 'No'), index=1)
        past_complaint = st.radio('Ever complaint?', ('Yes', 'No'), index=1)
        complaint_status = st.selectbox('Complaint Status', ('Solved',
                                                             'Solved in Follow-up',
                                                             'Unsolved',
                                                             'Not Applicable',
                                                             'Not Applicable'), index=1)
        feedback = st.selectbox('Feedback', ('Poor Product Quality',
                                             'Poor Website',
                                             'Poor Customer Service',
                                             'No reason specified',
                                             'Too many ads',
                                             'Reasonable Price',
                                             'User Friendly Website',
                                             'Products always in Stock',
                                             'Quality Customer Care'), index=1)
        st.markdown('---')

        submitted = st.form_submit_button('Predict')

    data_inf = {
        'user_id': user_id,
        'age': age,
        'gender': gender,
        'region_category': region_category,
        'membership_category': membership_category,
        'joining_date': joining_date,
        'joined_through_referral': joined_through_referral,
        'preferred_offer_types': preferred_offer_types,
        'medium_of_operation': medium_of_operation,
        'internet_option': internet_option,
        'last_visit_time': last_visit_time,
        'days_since_last_login': days_since_last_login,
        'avg_time_spent': avg_time_spent,
        'avg_transaction_value': avg_transaction_value,
        'avg_frequency_login_days': avg_frequency_login_days,
        'points_in_wallet': points_in_wallet,
        'used_special_discount': used_special_discount,
        'offer_application_preference': offer_application_preference,
        'past_complaint': past_complaint,
        'complaint_status': complaint_status,
        'feedback': feedback
    }

    # nama kolom dataframe yang dibuat harus sama dengan dataframe originial

    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    if submitted:
        # Preprocessing
        data_inf["joining_date"] = pd.to_datetime(data_inf["joining_date"])
        data_inf["last_visit_time"] = pd.to_datetime(
            data_inf["last_visit_time"])

        data_inf['last_visit_time'] = pd.to_datetime(
            data_inf['last_visit_time'], format='%H:%M').dt.hour
        data_inf['joining_year'] = pd.DatetimeIndex(
            data_inf['joining_date']).year
        data_inf['joining_month'] = pd.DatetimeIndex(
            data_inf['joining_date']).month

        data_inf['joining_month'] = data_inf['joining_month'].apply(
            lambda x: calendar.month_abbr[x])

        data_inf['joining_year'] = data_inf['joining_year'].astype(str)

        data_inf.drop(['user_id'], axis=1, inplace=True)
        data_inf.drop(['joining_date'], axis=1, inplace=True)
        data_inf.rename(columns={'churn_risk_score': 'churn'}, inplace=True)

        # Transform Inference-Set

        data_inf_transform = model_pipeline.transform(data_inf)
        data_inf_transform

        # Predict using Neural Network
        y_pred_inf = model_ann.predict(data_inf_transform)
        y_pred_inf = np.where(y_pred_inf >= 0.5, 1, 0)
        st.write('# Rating: ', str(int(y_pred_inf)))


# agar bisa di run sendiri, tanpa dari main
if __name__ == '__main__':
    run()
