import streamlit as st
import pandas as pd
import numpy as np
import base64
import pickle

#streamlit
def main():        
    
    st.set_page_config(page_title = 'COVID-19 Survivor',\
                       page_icon = 'logo_covid.jpg',
                       layout='wide',
                       initial_sidebar_state = 'expanded')
    
    c1, c2, c3 = st.columns([3,1,2])
    c1.title('Simulador - Sobrevivência do COVID-19')
    c2.image('logo_covid.jpg', width=300)
    c3.image('coronavirus.gif', width=150)
    with st.expander('Descrição do App',expanded=True):
        st.markdown('O objetivo principal desta ferramenta é realizar predições sobre a chance de um paciente sobreviver considerando as suas comorbidades e outras variáveis clinicas caso seja contaminado pelo COVID 19')
    
#################################################################################################################
    with st.sidebar:
        database = st.radio('fonte dos dados de entrada (X):',('Manual', 'CSV'))
        database = st.radio('fonte dos dados de entrada (X):',('Manual São Paulo', 'CSV 2'))
        
        if database == 'CSV':
            st.info('Upload do CSV')
            file = st.file_uploader('Selecione o arquivo CSV contendo as colunas acima descritas',type='São Paulo - csv')
            if file:
                Xtest = pd.read_csv(file)
                mdl_lgbm = pickle.load(open('pickle_mdl_lregression_select.pkl', 'rb'))
                ypred = mdl_lgbm.predict(Xtest)				    
            if file:
                Xtest = pd.read_csv(file)
                mdl_lgbm = pickle.load(open('pickle_mdl_lregression_select.pkl', 'rb'))
                ypred = mdl_lgbm.predict(Xtest)
        else:
            X1 = st.slider('Idade do paciente de 0 a 109 anos',0,109,step=1)
            X2 = st.sidebar.selectbox('Sexo do paciente? (0 - Masculino | 1 - Feminino)',(0,1))
            X3 = st.sidebar.selectbox('Paciente tem pneumonia nosocomial? (0 - Não | 1 - Sim)',(0,1,9))
            X4 = st.sidebar.selectbox('Paciente tem febre? (0 - Não | 1 - Sim)',(0,1))
            X5 = st.sidebar.selectbox('Paciente tem tosse? (0 - Não | 1 - Sim)',(0,1))
            X6 = st.sidebar.selectbox('Paciente tem dor de garganta? (0 - Não | 1 - Sim)',(0,1))
            X7 = st.sidebar.selectbox('Paciente tem falta de ar? (0 - Não | 1 - Sim)',(0,1))
            X8 = st.sidebar.selectbox('Paciente tem saturação baixa? (0 - Não | 1 - Sim)',(0,1))
            X9 = st.sidebar.selectbox('Paciente tem diarreia? (0 - Não | 1 - Sim)',(0,1))
            X10 = st.sidebar.selectbox('Paciente tem vômitos? (0 - Não | 1 - Sim)',(0,1))
            X11 = st.sidebar.selectbox('Paciente tem dor abdominal? (0 - Não | 1 - Sim)',(0,1))
            X12 = st.sidebar.selectbox('Paciente tem fadiga? (0 - Não | 1 - Sim)',(0,1))
            X13 = st.sidebar.selectbox('Paciente tem perda de olfato? (0 - Não | 1 - Sim)',(0,1))
            X14 = st.sidebar.selectbox('Paciente tem perda de paladar? (0 - Não | 1 - Sim)',(0,1))
            X15 = st.sidebar.selectbox('Paciente tem problemas cardíacos? (0 - Não | 1 - Sim)',(0,1))
            X16 = st.sidebar.selectbox('Paciente tem problemas sanguíneos (trombose, anemia, linfoma e leucemia)?  (0 - Não | 1 - Sim)',(0,1))
            X17 = st.sidebar.selectbox('Paciente tem síndrome de Down? (0 - Não | 1 - Sim)',(0,1))
            X18 = st.sidebar.selectbox('Paciente tem problemas no fígado? (0 - Não | 1 - Sim)',(0,1))
            X19 = st.sidebar.selectbox('Paciente tem asma? (0 - Não | 1 - Sim)',(0,1))
            X20 = st.sidebar.selectbox('Paciente tem diabetes? (0 - Não | 1 - Sim)',(0,1))
            X21 = st.sidebar.selectbox('Paciente tem problemas neurológicos? (0 - Não | 1 - Sim)',(0,1))
            X22 = st.sidebar.selectbox('Paciente tem problemas pulmonares? (0 - Não | 1 - Sim)',(0,1))
            X23 = st.sidebar.selectbox('Paciente tem imunodepressão (HIV/Câncer)? (0 - Não | 1 - Sim)',(0,1))
            X24 = st.sidebar.selectbox('Paciente tem problemas nos rins? (0 - Não | 1 - Sim)',(0,1))
            X25 = st.sidebar.selectbox('Paciente é obeso? (0 - Não | 1 - Sim)',(0,1))


            Xtest = pd.DataFrame({'IDADE_ANOS': [X1], 'CS_SEXO': [X2], 'NOSOCOMIAL': [X3], 'FEBRE': [X4], 
                                      'TOSSE': [X5], 'GARGANTA': [X6], 'DISPNEIA': [X7], 'SATURACAO': [X8], 
                                      'DIARREIA': [X9], 'VOMITO': [X10], 'DOR_ABD': [X11], 
                                      'FADIGA': [X12], 'PERD_OLFT': [X13], 'PERD_PALA': [X14],
                                      'CARDIOPATI': [X15], 'HEMATOLOGI': [X16], 'SIND_DOWN': [X17],
                                      'HEPATICA': [X18], 'ASMA': [X19], 'DIABETES': [X20],
                                      'NEUROLOGIC': [X21], 'PNEUMOPATI': [X22], 'IMUNODEPRE': [X23],
                                      'RENAL': [X24], 'OBESIDADE': [X25]})
            
            mdl_lgbm = pickle.load(open('pickle_mdl_lregression_select.pkl', 'rb'))
            ypred = mdl_lgbm.predict(Xtest)
                                     
##################################################################################################################

    if database == 'Manual':
        with st.expander('Visualizar Dados de Entrada', expanded = False):
                st.dataframe(Xtest)
        with st.expander('Visualizar Predição', expanded = True):
                if ypred==0:
                    st.error(ypred[0])
                else:
                    st.success(ypred[0])
                    
        if st.button('Baixar arquivo csv'):
            df_download = Xtest.copy()
            df_download['Response_pred'] = ypred
            st.dataframe(df_download)
            csv = df_download.to_csv(sep=',',decimal=',',index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
            

    else: #database == 'CSV'
        if file:
            with st.expander('Visualizar Dados de Entrada', expanded = False):
                st.dataframe(Xtest)
            with st.expander('Visualizar Predições', expanded = False):
                st.dataframe(ypred)            
            
            if st.button('Baixar arquivo csv'):
                df_download = Xtest.copy()
                df_download['Response_pred'] = ypred
                st.write(df_download.shape)
                st.dataframe(df_download)
                csv = df_download.to_csv(sep=',',decimal=',',index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)

if __name__ == '__main__':
	main()
