import streamlit as st
import math
import numpy as np

from load_and_predict_RF import predict_a_sample_RF
from load_and_predict_GB import predict_a_sample_GB

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Predicting Early-Stage Financial Distress in Vietnamese Firms: A Machine Learning Perspective',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :earth_americas: Predicting Early-Stage Financial Distress in Vietnamese Firms: A Machine Learning Perspective

This application predicts the financial state of a listed firm in Vietnam stock exchanges in the next 1 year. 
Please input the following financial indicators and the interest rates of the State Bank of Vietnam & U.S. FED for prediction.

The unit of financial indicators is billion VND. Default values are extracted from audited financial statement of SDP (SDP Joint Stock Company) in 2018.
The default interest rates are the average rate of 2018 (SBV: 6.25%, FED: 1.71%). That company was labeled as severe distressed on 2019.

The prediction results fall into one of the 3 following possibilities: Soundness (Active), Mild Distress and Severe Distress.
'''

# Add some spacing
''
''

st.title('Select a prediction model')
algorithm = st.selectbox(
    "Which trained model would you like to adopt?",
    ("Random Forest (RF)", "Gradient Boosting (GB)"),
)

st.title('Then enter the financial indicators of the firm (in billion VND)')
A01 = st.number_input('Current assets', value=428.0)
A05 = st.number_input('Inventories', value=109.)
A08 = st.number_input('Fixed assets', value=27.0)
A11 = st.number_input('Total assets', value=539.0)
A13 = st.number_input('Current liabilities', value=459.0)
A14 = st.number_input('Long-term liabilities', value=2.0)
A12 = st.number_input('Total liabilities', value=461.0)
B01 = st.number_input('Sales revenue', value=313.0)
B03 = st.number_input('Net income', value=16.0)
B11 = st.number_input('EBIT', value=-10.0)
A18 = st.number_input('Retained earnings', value=-68.0)
A15 = st.number_input('Equity', value=79.0)
SBV = st.number_input('Interest rate of the State Bank of Vietnam (%)', value=6.25)
FED = st.number_input('Interest rate of the U.S. Federal Reserve (%)', value=1.71)

X1 = A01 / A13
X2 = (A01 - A13) / A11
X3 = (A01 - A13) / B01
X4 = B11 / A11
X5 = B03 / A15
X6 = B03 / A11
X7 = B11 / B01
X8 = A18 / A11
X9 = B01 / A11
X10 = A13 / A11
X11 = A14 / A11
X12 = A12 / A11
X13 = A05 / (A01 - A13)
X14 = A14 / A01
X15 = math.log(A11)
X16 = math.log(B01)
X17 = A08 / A11
X18 = A15 / A11
X19 = A13 / A12

sample = [X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18, X19, SBV, FED]
sample = np.asarray(sample, dtype=np.float32)

if algorithm == 'Gradient Boosting (GB)':
    print('GB selected')
    pred = predict_a_sample_GB(sample)
else:
    print('RF selected')
    pred = predict_a_sample_RF(sample)

st.title('Calculated financial ratios: ')
st.text('Current ratio [current assets / current liabilities]: ' + "{:.2f}".format(X1))
st.text('WC / TA ratio [(current assets – current liabilities) / total assets]: ' + "{:.2f}".format(X2))
st.text('WC / S ratio [(current assets – current liabilities) / sales revenue]: ' + "{:.2f}".format(X3))
st.text('EBIT / TA ratio [EBIT / total assets]: ' + "{:.2f}".format(X4))
st.text('Return on equity [net income / total owner’s equity]: ' + "{:.2f}".format(X5))
st.text('Return on assets [net income / total assets]: ' + "{:.2f}".format(X6))
st.text('EBIT / S ratio [EBIT / sales revenue]: ' + "{:.2f}".format(X7))
st.text('RE / TA ratio [retained earnings / total assets]: ' + "{:.2f}".format(X8))
st.text('S / TA ratio [sales revenue / total assets]: ' + "{:.2f}".format(X9))
st.text('CL / TA ratio [current liabilities / total assets]: ' + "{:.2f}".format(X10))
st.text('LTL / TA ratio [long-term liabilities / total assets]: ' + "{:.2f}".format(X11))
st.text('TL / TA ratio [total liabilities / total assets]: ' + "{:.2f}".format(X12))
st.text('I / WC ratio [inventories / (current assets – current liabilities)]: ' + "{:.2f}".format(X13))
st.text('LTL / CA ratio [long-term liabilities / current assets]: ' + "{:.2f}".format(X14))
st.text('Natural logarithm of total assets: ' + "{:.2f}".format(X15))
st.text('Natural logarithm of sales: ' + "{:.2f}".format(X16))
st.text('FA / TA ratio [fixed assets / total assets]: ' + "{:.2f}".format(X17))
st.text('E / TA ratio [total owner’s equity / total assets]: ' + "{:.2f}".format(X18))
st.text('CL / TL ratio [current liabilities / total liabilities]: ' + "{:.2f}".format(X19))

if pred[0] == 0:
    st.title(algorithm + ' predicts: SOUNDNESS in the next year.')
elif pred[0] == 1:
    st.title(algorithm + ' predicts: MILD DISTRESS in the next year.')
else:
    st.title(algorithm + ' predicts: SEVERE DISTRESS in the next year.')
