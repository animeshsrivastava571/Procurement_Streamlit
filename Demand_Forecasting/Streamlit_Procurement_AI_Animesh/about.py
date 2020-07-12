import streamlit as st
from PIL import Image

def display_about():
    st.title('Procurement.AI')
    st.markdown('<style>h1{color: black;}</style>', unsafe_allow_html=True)
    st.text('Created by animesh.sr@hcl.com')
    img = Image.open('Img2.jpg')
    st.image(img,use_column_width=True)
    st.markdown('## About')

    st.markdown('''Procurement.AI is an initiative by COE Data Science at HCL to create an
    end to end ML based tool for Supply Chain in the domain of procurement'''
                )
    
    img = Image.open('Img3.PNG')
    st.image(img,width=800)
    st.markdown('### Limitations')
    st.markdown('* Supports only CSV as input')
    # st.markdown('* By default, it fetches all the files from the `JMETER_HOME` folder.')
    # st.markdown('* Limited number of charts has been added, other type of charts can be added by custom coding.')
    # st.markdown('* Doesn\'t support distributed load testing model.')
    #
    # st.markdown('### Known Issues')
    # st.markdown('* Doesn\'t execute if the JMeter file name which has space')
    # st.markdown('* Quick Navigation between Execute and Analyze may break the code, you may need to launch the app again.')
    # st.markdown('* Doesn\'t display the JMeter test results runtime')


def display_sidebar():
    st.sidebar.markdown('---')
    
    st.sidebar.title('Disclaimer')
    st.sidebar.info('This is a WIP product by the COE Data Science and subsequent features will be added')
    st.sidebar.title('About')
    st.sidebar.info('''This app has been developed by the [COE Data Science team](https://hcl.com)
                    'using [Python](https://www.python.org/),
    [Streamlit](https://streamlit.io/), and [Plotly](https://plotly.com/python/)
                    ''')
