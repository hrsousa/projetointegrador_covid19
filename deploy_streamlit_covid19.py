import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt
from plotly import graph_objs as go
import numpy as np 
import base64
import pickle

st.title("Simulador - Sobrevivência do COVID-19")
st.image("logo_covid.jpg",width = 550)
nav = st.sidebar.radio("Navegação",["Home","Predição","Gráficos"])

if nav == "Home":
    st.markdown("O objetivo principal desta ferramenta é realizar predições sobre a chance de um paciente sobreviver considerando as suas comorbidades caso seja contaminado pelo COVID 19")
    st.title("Projeto Integrador COVID 19 - 2021")
    st.title("Curso Data Science - Digital House")
    
if nav == "Predição":
    
    with st.sidebar:
        st.sidebar.header('Entrada de dados do usuário:')
        st.sidebar.markdown("""[Exemplo de arquivo CSV](https://raw.githubusercontent.com/hrsousa/projetointegrador_covid19/main/exemplo_dados.csv)""")
        database = st.radio('Seleção da fonte dos dados de entrada (X):',('Manual', 'CSV'))        
        if database == 'CSV':
            st.info('Upload do CSV')
            file = st.file_uploader('Selecione o arquivo CSV contendo as colunas acima descritas',type='csv')
            if file:
                Xtest = pd.read_csv(file)
                mdl_lgbm = pickle.load(open('pickle_mdl_lregression_select.pkl', 'rb'))
                ypred = mdl_lgbm.predict(Xtest)				    
            if file:
                Xtest = pd.read_csv(file)
                mdl_lgbm = pickle.load(open('pickle_mdl_lregression_select.pkl', 'rb'))
                ypred = mdl_lgbm.predict(Xtest)
        else:
            X1 = st.slider('1. Idade do paciente de 0 a 109 anos',0,109,step=1)
            X2 = st.sidebar.selectbox('2. Qual o sexo do paciente? (0 - Masculino | 1 - Feminino)',(0,1))
            X3 = st.sidebar.selectbox('3. Paciente gestante? (Idade gestacional da paciente) (0 - Não | 1 - Sim)',(0,1))
            X4 = st.sidebar.selectbox('4. Cor ou raça declarada pelo paciente',(1,2,3,4,5,9))
            X5 = st.sidebar.selectbox('5. Nível de escolaridade do paciente',(0,1,2,3,4,5,9))
            X6 = st.sidebar.selectbox('6. Tem histórico de viagem internacional nos últimos 14 dias?',(1, 2, 9))
            X7 = st.sidebar.selectbox('7. É caso proveniente de surto de Síndrome Gripal que evoluiu para SRAG? (0 - Não | 1 - Sim)',(0,1))
            X8 = st.sidebar.selectbox('8. Trata-se de caso nosocomial (infecção adquirida no hospital)? (0 - Não | 1 - Sim)',(0,1))
            X9 = st.sidebar.selectbox('9. Paciente trabalha ou tem contato direto com aves, suínos, ou outro animal? (0 - Não | 1 - Sim)',(0,1))
            X10 = st.sidebar.selectbox('10. Paciente apresentou febre? (0 - Não | 1 - Sim)',(0,1))
            X11 = st.sidebar.selectbox('11. Paciente apresentou tosse? (0 - Não | 1 - Sim)',(0,1))
            X12 = st.sidebar.selectbox('12. Paciente apresentou dor de garganta? (0 - Não | 1 - Sim)',(0,1))
            X13 = st.sidebar.selectbox('13. Paciente apresentou dispneia (falta de ar)? (0 - Não | 1 - Sim)',(0,1))
            X14 = st.sidebar.selectbox('14. Paciente apresentou desconforto respiratório? (0 - Não | 1 - Sim)',(0,1))
            X15 = st.sidebar.selectbox('15. Paciente apresentou saturação O2 menor que 95%? (0 - Não | 1 - Sim)',(0,1))
            X16 = st.sidebar.selectbox('16. Paciente apresentou diarreia?  (0 - Não | 1 - Sim)',(0,1))
            X17 = st.sidebar.selectbox('17. Paciente apresentou vômitos? (0 - Não | 1 - Sim)',(0,1))
            X18 = st.sidebar.selectbox('18. Paciente apresentou dor abdominal? (0 - Não | 1 - Sim)',(0,1))
            X19 = st.sidebar.selectbox('19. Paciente apresentou fadiga? (0 - Não | 1 - Sim)',(0,1))
            X20 = st.sidebar.selectbox('20. Paciente apresentou perda de olfato? (0 - Não | 1 - Sim)',(0,1))
            X21 = st.sidebar.selectbox('21. Paciente apresentou perda de paladar? (0 - Não | 1 - Sim)',(0,1))
            X22 = st.sidebar.selectbox('22. Paciente apresentou outro(s) sintoma(s)? (0 - Não | 1 - Sim)',(0,1))
            X23 = st.sidebar.selectbox('23. Paciente apresenta algum fator de risco? (0 - Não | 1 - Sim)',(0,1))
            X24 = st.sidebar.selectbox('24. Paciente é puérpera ou parturiente (45 dias pós parto)? (0 - Não | 1 - Sim)',(0,1))
            X25 = st.sidebar.selectbox('25. Paciente possui Doença Cardiovascular Crônica? (0 - Não | 1 - Sim)',(0,1))
            X26 = st.sidebar.selectbox('26. Paciente possui Doença Hematológica Crônica? (0 - Não | 1 - Sim)',(0,1))
            X27 = st.sidebar.selectbox('27. Paciente possui síndrome de Down? (0 - Não | 1 - Sim)',(0,1))
            X28 = st.sidebar.selectbox('28. Paciente possui Doença Hepática Crônica? (0 - Não | 1 - Sim)',(0,1))
            X29 = st.sidebar.selectbox('29. Paciente possui asma? (0 - Não | 1 - Sim)',(0,1))
            X30 = st.sidebar.selectbox('30. Paciente possui diabetes? (0 - Não | 1 - Sim)',(0,1))
            X31 = st.sidebar.selectbox('31. Paciente possui Doença Neurológica? (0 - Não | 1 - Sim)',(0,1))
            X32 = st.sidebar.selectbox('32. Paciente possui outra pneumopatia crônica? (0 - Não | 1 - Sim)',(0,1))
            X33 = st.sidebar.selectbox('33. Paciente possui Imunodeficiência ou Imunodepressão? (0 - Não | 1 - Sim)',(0,1))
            X34 = st.sidebar.selectbox('34. Paciente possui Doença Renal Crônica?? (0 - Não | 1 - Sim)',(0,1))
            X35 = st.sidebar.selectbox('35. Paciente possui obesidade? (0 - Não | 1 - Sim)',(0,1))
            X36 = st.sidebar.selectbox('36. Paciente possui outro(s) fator(es) de risco?(0 - Não | 1 - Sim)',(0,1))
            X37 = st.sidebar.selectbox('37. Fez uso de antiviral para gripe? (0 - Não | 1 - Sim)',(0,1))
            X38 = st.sidebar.selectbox('38. Paciente foi internado em UTI? (0 - Não | 1 - Sim)',(0,1))
            X39 = st.sidebar.selectbox('39. Paciente fez uso de suporte ventilatório (ventilação mecânica)?',(0,1,2))
            X40 = st.sidebar.selectbox('40. Informar resultado de Raio X de tórax',(1, 2, 3, 4, 5, 6))
            X41 = st.sidebar.selectbox('41. Informar resultado da tomografia',(1, 2, 3, 4, 5, 6))
            X42 = st.sidebar.selectbox('42. Resultado do teste de RT-PCR/outro método por Biologia Molecular',(1, 2, 3, 4, 5, 9))
            X43 = st.sidebar.selectbox('43. Resultado da Sorologia para SARS-CoV-2 (IgA)',(1, 2, 3, 4, 5))
            X44 = st.sidebar.selectbox('44. Resultado da Sorologia para SARS-CoV-2 (IgG)',(1, 2, 3, 4, 5))
            X45 = st.sidebar.selectbox('45. Resultado da Sorologia para SARS-CoV-2 (IgM)',(1, 2, 3, 4, 5))


            Xtest = pd.DataFrame({'IDADE_ANOS': [X1], 'CS_SEXO': [X2], 'CS_GESTANT': [X3], 'CS_RACA': [X4], 
                                      'CS_ESCOL_N': [X5], 'HISTO_VGM': [X6], 'SURTO_SG': [X7], 'NOSOCOMIAL': [X8], 
                                      'AVE_SUINO': [X9], 'FEBRE': [X10], 'TOSSE': [X11], 
                                      'GARGANTA': [X12], 'DISPNEIA': [X13], 'DESC_RESP': [X14],
                                      'SATURACAO': [X15], 'DIARREIA': [X16], 'VOMITO': [X17],
                                      'DOR_ABD': [X18], 'FADIGA': [X19], 'PERD_OLFT': [X20],
                                      'PERD_PALA': [X21], 'OUTRO_SIN': [X22], 'FATOR_RISC': [X23],
                                      'PUERPERA': [X24], 'CARDIOPATI': [X25], 'HEMATOLOGI': [X26],
                                      'SIND_DOWN': [X27], 'HEPATICA': [X28], 'ASMA': [X29],
                                      'DIABETES': [X30], 'NEUROLOGIC': [X31], 'PNEUMOPATI': [X32],
                                      'IMUNODEPRE': [X33], 'RENAL': [X34], 'OBESIDADE': [X35],
                                      'OUT_MORBI': [X36], 'ANTIVIRAL': [X37], 'UTI': [X38],
                                      'SUPORT_VEN': [X39], 'RAIOX_RES': [X40], 'TOMO_RES': [X41],
                                      'PCR_RESUL': [X42], 'RES_IGA': [X43], 'RES_IGG': [X44],
                                      'RES_IGM': [X45]})
            
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

if nav == "Gráficos":
    st.header("Plotagem de gráficos da pandemia (em construção...)")
    