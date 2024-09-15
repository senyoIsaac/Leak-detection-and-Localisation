import joblib
import streamlit as st
import pandas as np
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
from streamlit_image_select import image_select
import time
from playsound import playsound
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Load your trained model
# model = ... (load your model here)

# Load the beep sound
beep_sound = 'beep-04.mp3'
pygame.mixer.music.load(beep_sound)

testPressure = [
    [1.49816048, 3.80285723, 2.92797577, 2.39463394, 0.62407456],
    [0.62397808, 0.23233445, 3.46470458, 2.40446005, 2.83229031],
    [0.08233798, 3.87963941, 3.32977056, 0.84935644, 0.72729987],
    [0.73361804, 1.21696897, 2.09902573, 1.72778007, 1.16491656],
    [2.44741158, 0.55797544, 1.16857859, 1.46544737, 1.82427994],
    [3.14070385, 0.79869513, 2.05693775, 2.36965828, 0.18580165],
    [2.43017941, 0.68209649, 0.26020637, 3.79554215, 3.86252813],
    [3.23358939, 1.21845508, 0.39068846, 2.73693211, 1.76060997],
    [0.48815294, 1.98070764, 0.13755408, 3.63728161, 1.03511993],
    [2.65008914, 1.2468443, 2.08027208, 2.18684112, 0.73941782],
    [3.87833851, 3.10053129, 3.75799577, 3.5793094, 2.39159992],
    [3.68749694, 0.35397001, 0.78393145, 0.18090916, 1.30132132],
    [1.55470916, 1.08539613, 3.31495004, 1.42701331, 1.12373804],
    [2.17078433, 0.5636969, 3.20878792, 0.29820257, 3.94754775],
    [3.08897908, 0.79486273, 0.02208847, 3.26184571, 2.82742938],
    [2.91602867, 3.08508139, 0.29617861, 1.43386291, 0.46347624],
    [3.4524137, 2.49319251, 1.3235921, 0.2542334, 1.24392929],
    [1.30073329, 2.91842471, 2.55022989, 3.54885097, 1.8888597],
    [0.47837698, 2.85297915, 3.04314019, 2.24510879, 3.08386872],
    [1.97518239, 2.09093132, 1.71016407, 0.10167651, 0.43156571],
    [0.12571674, 2.54564165, 1.25742392, 2.03428276, 3.6302659],
    [0.99716892, 1.64153169, 3.02220455, 0.91519266, 0.30791964],
    [1.15900581, 0.64488515, 3.71879061, 3.23248152, 2.53361503],
    [3.48584236, 3.21468831, 0.74628024, 3.57023599, 2.15736897],
    [3.22976062, 3.5843652, 1.2720139, 0.4402077, 0.91174065],
    [1.70843115, 3.27205906, 3.44292233, 0.02780852, 2.04298921],
    [1.66964401, 0.88843124, 0.47946147, 1.35046069, 3.77163882],
    [1.29281173, 2.07516249, 2.81207584, 1.45451841, 3.88712833],
    [3.84978918, 1.00712918, 1.98899402, 1.20351324, 1.13936198],
    [0.14754779, 2.43825734, 2.01071609, 0.205915, 1.11458586],
    [3.63306354, 0.95824756, 0.57957949, 1.95781104, 3.94260182],
    [0.96822109, 2.68854219, 3.04647846, 0.95055018, 2.91286539],
    [1.47113253, 2.52922332, 2.53411884, 2.14309874, 0.36115908],
    [3.34120998, 1.28312026, 0.74607404, 0.16310057, 2.36357177],
    [2.71025745, 0.06635132, 2.04837223, 0.9059831, 2.58069116],
    [0.69746572, 2.76375095, 1.54694139, 3.74691995, 0.55008378],
    [1.3642654, 0.45389408, 3.69877447, 3.50935741, 1.03176651],
    [2.63993618, 3.2688888, 2.22080325, 2.11860231, 0.96740916],
    [0.37241107, 3.58886303, 3.60167223, 2.53240583, 1.35611916],
    [1.3968383, 2.90382272, 3.58844104, 3.5483457, 3.11950218],
    [2.56812658, 0.33655986, 0.64651486, 3.59421675, 2.42571624],
    [0.03678821, 0.40588617, 2.65400708, 0.02024634, 0.64323221],
    [2.19493516, 2.76758079, 2.60784504, 0.89707724, 2.84871689],
    [0.94899635, 1.30159879, 2.98596562, 2.5985316, 3.39689364],
    [2.63045157, 2.27323441, 0.37469907, 1.47086321, 1.06080947],
    [0.97595857, 3.89204222, 1.5723909, 3.56818622, 2.5245545],
    [3.17924521, 2.01054837, 2.30761554, 1.97007078, 0.78097195],
    [2.88980846, 1.12308945, 0.09726387, 2.58188918, 0.70844272],
    [3.76183434, 3.81571431, 3.65945756, 1.4806348, 0.06182647],
    [3.71327425, 1.71273659, 3.86661928, 3.85447991, 3.41203782]
]


st.set_page_config(page_title="Pipe Network Leakage Detection",layout="wide")
model = joblib.load("leak_classifier.joblib")

with st.sidebar:
     st.image('Faucet.jpeg', caption='Team Aqua AI')

selected = option_menu(
        menu_title=None,
        options=["Home","Monitor Leak","Monitor Flow","Leak History","ML Module"],
        orientation='horizontal',
        styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "15px"},
                "nav-link": {
                    "font-size": "17px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "black"},
            },
    )

if selected == "Home":
    st.subheader("Pipe Network Leakage Detection")
    column1,column2 = st.columns([5,1])
    with column1:
        st.image('setup.jpg', caption='Our Setup')
        st.image('Animation.gif', caption='Team Aqua AI')
        #st.video(data="SystemAnimation.gif")

if selected == "Monitor Leak":

    st.title("Real-time Leak Prediction")

    def predictLeak(p1, p2, p3, p4, p5):
        new_set = np.array([p1, p2, p3, p4, p5]).reshape(1, -1)
        new_set_prediction = model.predict(new_set)

        if new_set_prediction == 1:
            st.write("Leak", new_set_prediction)
            pygame.mixer.music.play()  # Play beep sound on leak detection
        else:
            st.write("No Leak", new_set_prediction)
    # Display each row of the testPressure data array
    for i, row in enumerate(testPressure):
        st.write(f"Pressure Reading {i+1}: {row}")
        time.sleep(1)
        predictLeak(*row) 
        time.sleep(5)  # Adding a delay for real-time effect

if selected == "ML Module":
    column1,column2 = st.columns([2,2])
    with column1:
        df = pd.read_excel("./PressureOutput.xlsx",sheet_name="Pressure")
        st.dataframe(df)
    with column2:
        st.image('./Visualizations/Confusion Matrix.jpg')

    st.markdown("""---""")

    column3, column4, column44 = st.columns([1,3,3])

    with column3:
        P1 = st.number_input(label="Pressure J1")
        P2 =  st.number_input(label="Pressure J2")
        P3 = st.number_input(label="Pressure J3")
        P4 = st.number_input(label="Pressure J4")
        P5 = st.number_input(label="Pressure J5")

    with column44:
        column5,column6 = st.columns([1,10])
        with column6:
        
            def predictLeak(p1,p2,p3,p4,p5):
                new_set = np.array([p1,p2,p3,p4,p5]).reshape(1, -1)
                #scaler = StandardScaler().fit(x_train)
                #new_set_scaled = scaler.transform(new_set)
                new_set_prediction = model.predict(new_set)
                if new_set_prediction == 1:
                    st.write(new_set_prediction)
                    #print("Leak", new_set_prediction)
                else:
                    st.write(new_set_prediction)
                    print("No Leak", new_set_prediction)

            predictLeak(P1,P2,P3,P4,P5)
                

       

if selected == "Pressure":
    st.subheader("Pressure Monitor")
    column1,column2 = st.columns(2)
    with column2:

        Junction1,Junction2,Junction3,Junction4 = st.columns(4)

        with Junction1:
            st.info('Junction 2')
            st.metric(label='Pressure',value="5") #Update arduino-read pressure here
            
        with Junction2:
            st.info('Junction 2')
            st.metric(label='Pressure',value="5") #Update arduino-read pressure here

        
        with Junction3:
            st.info('Junction 2')
            st.metric(label='Pressure',value="5") #Update arduino-read pressure here

        with Junction4:
            st.info('Junction 2')
            st.metric(label='Pressure',value="5") #Update arduino-read pressure here   

    st.markdown("""---""")
        

#st.sidebar.image('Faucet.jpeg',caption="Water Leak")
#option_menu(menu_title="Monitor",options=options,orientation='horizontal')
