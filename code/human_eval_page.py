import streamlit as st
import pandas as pd
import html
import os 
import json
from IPython.display import display, HTML
#from IPython.core.display import display, HTML

def underline_words_in_red(text, words_to_underline):
    """ Underline words that are present in words_to_underline with a red underline. """
    for word in words_to_underline:
        text = text.replace(word, f'<u style="color: black;">{word}</u>')
    return text

def visualize_importance(input_items, normalized_importance, words_to_underline=[]):
    """ General function to visualize importance for any granularity - word, token, sentence. """
    max_alpha = 0.5
    highlighted_text = []

    for i in range(len(input_items)):
        item = input_items[i]
        weight = normalized_importance[i]
        item = item.replace('Ġ', '').replace('▁', '') # 'Ġ' roberta; '▁' T5
        if weight is not None:
            highlighted_item = f'<span style="background-color:rgba(135,206,250,{weight / max_alpha});">{item}</span>'
        else:
            highlighted_item = item
        highlighted_text.append(highlighted_item)

    combined_text = ' '.join(highlighted_text)
    combined_text = underline_words_in_red(combined_text, words_to_underline)
    #display(HTML(combined_text))
    return combined_text

def update_or_insert_data(data, json_file_path = '../data/result_tmp.json'):
    # Check if the JSON file exists
    if os.path.exists(json_file_path):
        # Read existing data
        with open(json_file_path, 'r') as file:
            try:
                file_data = json.load(file)
            except json.JSONDecodeError:
                file_data = {}
    else:
        file_data = {}
    
    # Update data if document_id exists or add new document_id
    file_data[data['document_id']] = data

    # Write the updated data back to the JSON file
    with open(json_file_path, 'w') as file:
        json.dump(file_data, file, indent=4)

def get_wis_html(df, index, model_name):
    word_list = df.iloc[index]['{}_word_list'.format(model_name)]
    word_imp_scores = df.iloc[index]['{}_wis'.format(model_name)]
    rlf_word = df.iloc[index]['rlf']
    #print(word_imp_scores)
    res = visualize_importance(word_list, word_imp_scores, words_to_underline=[rlf_word])
    return res

def get_visualize_content(df, index):
    document_id = df.iloc[index]['document_id']
    rlf_sent = df.iloc[index]['rlf_sent']
    no_rlf_sent = df.iloc[index]['rand_sent']
    llama2_wis_html = get_wis_html(df, index, 'llama2')
    roberta_wis_html = get_wis_html(df, index, 'roberta')
    gpt2_wis_html = get_wis_html(df, index, 'gpt2')
    t5_wis_html = get_wis_html(df, index, 't5')
    gpt4_wis_html = get_wis_html(df, index, 'gpt4')
    return [document_id, rlf_sent, no_rlf_sent, llama2_wis_html, roberta_wis_html, gpt2_wis_html, t5_wis_html, gpt4_wis_html]

if 'index' not in st.session_state:
    st.session_state['index'] = 0
    print('init index', st.session_state['index'])
    json_file_path = '../data/result_tmp.json'
    df_tmp = pd.read_parquet('../data/human_eval_sample.parquet') 
    st.session_state['df_tmp'] = df_tmp
    
[document_id, rlf_sent, no_rlf_sent, llama2_wis_html, roberta_wis_html, gpt2_wis_html, t5_wis_html, gpt4_wis_html] = get_visualize_content(st.session_state['df_tmp'], st.session_state['index'])
sample_size = st.session_state['df_tmp'].shape[0]
    
st.header(':pencil: Human Evaluation Page for RLF :student:', divider='rainbow')
# st.subheader('Annotation for Sentiment Label 	:smiley_cat: 	:smirk_cat:')
st.markdown('Document ID: **{}**'.format(document_id))
st.subheader('Give a binary sentiment label (Positive or Negative) for each sentence', divider='rainbow')

rlf_sent_label = st.radio(
    "# **{}**".format(rlf_sent),
    ["Positive", "Negative"],
    horizontal=True,
    label_visibility="visible")

no_rlf_sent_label = st.radio(
    "# **{}**".format(no_rlf_sent),
    ["Positive", "Negative"],
    horizontal=True,
    label_visibility="visible")


st.subheader('Give a reliability score for each result (Agree or Disagree).', divider='rainbow')

col1, col2 = st.columns([1,1])
with col1:
    st.markdown(llama2_wis_html, unsafe_allow_html=True)
     
with col2:
    llama2_rs = st.radio(
    "**llama2**",
    ["Agree", "Disagree"],
    horizontal=True,
    label_visibility="hidden")
    
col1, col2 = st.columns([1,1])
with col1:
    st.markdown(roberta_wis_html, unsafe_allow_html=True)
with col2:
    roberta_rs = st.radio(
    "**roberta**",
    ["Agree", "Disagree"],
    horizontal=True,
    label_visibility="hidden")
    
col1, col2 = st.columns([1,1])
with col1:
    st.markdown(gpt2_wis_html, unsafe_allow_html=True)
with col2:
    gpt2_rs = st.radio(
    "**gpt2**",
    ["Agree", "Disagree"],
    horizontal=True,
    label_visibility="hidden")
  
col1, col2 = st.columns([1,1])
with col1:
    st.markdown(t5_wis_html, unsafe_allow_html=True)
with col2:  
    t5_rs = st.radio(
    "**t5**",
    ["Agree", "Disagree"],
    horizontal=True,
    label_visibility="hidden")
    
col1, col2 = st.columns([1,1])
with col1:
    st.markdown(gpt4_wis_html, unsafe_allow_html=True)
with col2: 
    gpt4_rs = st.radio(
    "**gpt4**",
    ["Agree", "Disagree"],
    horizontal=True,
    label_visibility="hidden")
    
col1, col2, col3 = st.columns([1,1,1])  
with col1:
    if st.button("Prev", type="primary"):
        st.session_state['index'] -= 1
        if st.session_state['index'] <= 0:
            st.session_state['index'] = 0
        st.experimental_rerun()
        print('index - 1', st.session_state['index'])
                        
with col3: 
    if st.button("Next", type="primary"):
        data = {
            'index': st.session_state['index'],
            'document_id': document_id,
            'rlf_sent_label': rlf_sent_label,
            'no_rlf_sent_label': no_rlf_sent_label,
            'llama2_rs': llama2_rs,
            'roberta_rs': roberta_rs,
            'gpt3_rs': gpt2_rs,
            't5_rs': t5_rs,
            'gpt4_rs': gpt4_rs    
        } 
        update_or_insert_data(data)
        st.session_state['index'] += 1
        if st.session_state['index'] >= sample_size-1:
            st.session_state['index'] = sample_size-1
        st.experimental_rerun()
        print('index + 1', st.session_state['index'])
        
progress_text = "Your progress.... {}%".format(int(st.session_state['index']/(sample_size-1) * 100))
my_bar = st.progress(0, text=progress_text)
my_bar.progress(st.session_state['index']/(sample_size-1), text=progress_text)
    
    
    
    
    