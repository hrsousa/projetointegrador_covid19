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
    
    c1, c2 = st.columns([3,1])
    c1.title('Simulador - Sobrevivência do COVID-19')
    c2.image('logo_covid.jpg', width=300)
    with st.expander('Descrição do App',expanded=True):
        st.markdown('O objetivo principal desta ferramenta é realizar predições sobre a chance de um paciente sobreviver considerando as suas comorbidades e outras variáveis clinicas caso seja contaminado pelo COVID 19')
    
#################################################################################################################
    with st.sidebar:
        database = st.radio('fonte dos dados de entrada (X):',('Manual', 'CSV'))
        
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
            X2 = st.sidebar.selectbox('Qual o sexo do paciente? (0 - Masculino | 1 - Feminino)',(0,1))
            X3 = st.sidebar.selectbox('Paciente gestante? (Idade gestacional da paciente) (0 - Não | 1 - Sim)',(0,1))
            X4 = st.sidebar.selectbox('Cor ou raça declarada pelo paciente (0 - Não | 1 - Sim)',(0,1))
            X5 = st.sidebar.selectbox('Nível de escolaridade do paciente (0 - Não | 1 - Sim)',(0,1))
            X6 = st.sidebar.selectbox('Tem histórico de viagem internacional nos últimos 14 dias? (0 - Não | 1 - Sim)',(0,1))
            X7 = st.sidebar.selectbox('É caso proveniente de surto de Síndrome Gripal que evoluiu para Síndrome Respiratória Aguda Grave? (0 - Não | 1 - Sim)',(0,1))
            X8 = st.sidebar.selectbox('Trata-se de caso nosocomial (infecção adquirida no hospital)? Caso de SRAG com infecção adquirida após internação? (0 - Não | 1 - Sim)',(0,1))
            X9 = st.sidebar.selectbox('Paciente trabalha ou tem contato direto com aves, suínos, ou outro animal? (0 - Não | 1 - Sim)',(0,1))
            X10 = st.sidebar.selectbox('Paciente apresentou febre? (0 - Não | 1 - Sim)',(0,1))
            X11 = st.sidebar.selectbox('Paciente apresentou tosse? (0 - Não | 1 - Sim)',(0,1))
            X12 = st.sidebar.selectbox('Paciente apresentou dor de garganta? (0 - Não | 1 - Sim)',(0,1))
            X13 = st.sidebar.selectbox('Paciente apresentou dispneia (falta de ar)? (0 - Não | 1 - Sim)',(0,1))
            X14 = st.sidebar.selectbox('Paciente apresentou desconforto respiratório? (0 - Não | 1 - Sim)',(0,1))
            X15 = st.sidebar.selectbox('Paciente apresentou saturação O2 menor que 95%? (0 - Não | 1 - Sim)',(0,1))
            X16 = st.sidebar.selectbox('Paciente apresentou diarreia?  (0 - Não | 1 - Sim)',(0,1))
            X17 = st.sidebar.selectbox('Paciente apresentou vômitos? (0 - Não | 1 - Sim)',(0,1))
            X18 = st.sidebar.selectbox('Paciente apresentou dor abdominal? (0 - Não | 1 - Sim)',(0,1))
            X19 = st.sidebar.selectbox('Paciente apresentou fadiga? (0 - Não | 1 - Sim)',(0,1))
            X20 = st.sidebar.selectbox('Paciente apresentou perda de olfato? (0 - Não | 1 - Sim)',(0,1))
            X21 = st.sidebar.selectbox('Paciente apresentou perda de paladar? (0 - Não | 1 - Sim)',(0,1))
            X22 = st.sidebar.selectbox('Paciente apresentou outro(s) sintoma(s)? (0 - Não | 1 - Sim)',(0,1))
            X23 = st.sidebar.selectbox('Paciente apresenta algum fator de risco? (0 - Não | 1 - Sim)',(0,1))
            X24 = st.sidebar.selectbox('Paciente é puérpera ou parturiente (mulher que pariu recentemente – até 45 dias do parto)? (0 - Não | 1 - Sim)',(0,1))
            X25 = st.sidebar.selectbox('Paciente possui Doença Cardiovascular Crônica? (0 - Não | 1 - Sim)',(0,1))
            X26 = st.sidebar.selectbox('Paciente possui Doença Hematológica Crônica? (0 - Não | 1 - Sim)',(0,1))
            X27 = st.sidebar.selectbox('Paciente possui síndrome de Down? (0 - Não | 1 - Sim)',(0,1))
            X28 = st.sidebar.selectbox('Paciente possui Doença Hepática Crônica? (0 - Não | 1 - Sim)',(0,1))
            X29 = st.sidebar.selectbox('Paciente possui asma? (0 - Não | 1 - Sim)',(0,1))
            X30 = st.sidebar.selectbox('Paciente possui diabetes? (0 - Não | 1 - Sim)',(0,1))
            X31 = st.sidebar.selectbox('Paciente possui Doença Neurológica? (0 - Não | 1 - Sim)',(0,1))
            X32 = st.sidebar.selectbox('Paciente possui outra pneumopatia crônica? (0 - Não | 1 - Sim)',(0,1))
            X33 = st.sidebar.selectbox('Paciente possui Imunodeficiência ou Imunodepressão (diminuição da função do sistema imunológico)? (0 - Não | 1 - Sim)',(0,1))
            X34 = st.sidebar.selectbox('Paciente possui Doença Renal Crônica?? (0 - Não | 1 - Sim)',(0,1))
            X35 = st.sidebar.selectbox('Paciente possui obesidade? (0 - Não | 1 - Sim)',(0,1))
            X36 = st.sidebar.selectbox('Paciente possui outro(s) fator(es) de risco?(0 - Não | 1 - Sim)',(0,1))
            X37 = st.sidebar.selectbox('Fez uso de antiviral para gripe? (0 - Não | 1 - Sim)',(0,1))
            X38 = st.sidebar.selectbox('Paciente foi internado em UTI? (0 - Não | 1 - Sim)',(0,1))
            X39 = st.sidebar.selectbox('Paciente fez uso de suporte ventilatório (ventilação mecânica)? (0 - Não | 1 - Sim)',(0,1))
            X40 = st.sidebar.selectbox('Informar resultado de Raio X de tórax (0 - Não | 1 - Sim)',(0,1))
            X41 = st.sidebar.selectbox('Informar resultado da tomografia (0 - Não | 1 - Sim)',(0,1))
            X42 = st.sidebar.selectbox('Resultado do teste de RT-PCR/outro método por Biologia Molecular (0 - Não | 1 - Sim)',(0,1))
            X43 = st.sidebar.selectbox('Resultado da Sorologia para SARS-CoV-2 (IgA) (0 - Não | 1 - Sim)',(0,1))
            X44 = st.sidebar.selectbox('Resultado da Sorologia para SARS-CoV-2 (IgG) (0 - Não | 1 - Sim)',(0,1))
            X45 = st.sidebar.selectbox('Resultado da Sorologia para SARS-CoV-2 (IgM) (0 - Não | 1 - Sim)',(0,1))


            Xtest = pd.DataFrame({'IDADE_ANOS': [X1], 'CS_SEXO': [X2], 'CS_GESTANT': [X3], 'CS_RACA': [X4], 
                                      'CS_ESCOL_N': [X5], 'HISTO_VGM': [X6], 'SURTO_SG': [X7], 'NOSOCOMIAL': [X8], 
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
