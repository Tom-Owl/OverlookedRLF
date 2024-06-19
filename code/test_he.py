import streamlit as st
import pandas as pd
import os
import json

# Assuming other functions are defined correctly above
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

# Load Data
df_tmp = pd.read_parquet('../data/human_eval_sample.parquet')
json_file_path = '../data/result_tmp.json'


def update_or_insert_data(data):
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


# Initialize or restore session state
if 'index' not in st.session_state:
    st.session_state['index'] = 0

# Function to update session state based on index
def update_session_content(index):
    content = get_visualize_content(df_tmp, index)
    for key, value in zip(['document_id', 'rlf_sent', 'no_rlf_sent', 'llama2_wis_html', 'roberta_wis_html', 'gpt2_wis_html', 't5_wis_html', 'gpt4_wis_html'], content):
        st.session_state[key] = value

# Initialize session state content
if 'document_id' not in st.session_state:
    update_session_content(st.session_state['index'])

st.header(':pencil: Human Evaluation Page for RLF :student:', divider='rainbow')

st.markdown(f'Document ID: **{st.session_state["document_id"]}**')
st.subheader('Give a binary sentiment label (Positive or Negative) for each sentence', divider='rainbow')

rlf_sent_label = st.radio(
    f"# **{st.session_state['rlf_sent']}**",
    ["Positive", "Negative"],
    horizontal=True,
    label_visibility="visible",
    key='rlf_sent_label')

no_rlf_sent_label = st.radio(
    f"# **{st.session_state['no_rlf_sent']}**",
    ["Positive", "Negative"],
    horizontal=True,
    label_visibility="visible",
    key='no_rlf_sent_label')

# Display visual content for models using markdown
st.markdown(st.session_state['llama2_wis_html'], unsafe_allow_html=True)
st.markdown(st.session_state['roberta_wis_html'], unsafe_allow_html=True)
st.markdown(st.session_state['gpt2_wis_html'], unsafe_allow_html=True)
st.markdown(st.session_state['t5_wis_html'], unsafe_allow_html=True)
st.markdown(st.session_state['gpt4_wis_html'], unsafe_allow_html=True)

# Navigation buttons
prev_button, next_button = st.columns(2)
with prev_button:
    if st.button("Prev"):
        new_index = max(0, st.session_state['index'] - 1)
        st.session_state['index'] = new_index
        update_session_content(new_index)

with next_button:
    if st.button("Next"):
        new_index = min(df_tmp.shape[0] - 1, st.session_state['index'] + 1)
        st.session_state['index'] = new_index
        update_session_content(new_index)

# Display progress
progress_percent = int(st.session_state['index'] / (df_tmp.shape[0] - 1) * 100)
st.progress(progress_percent)
st.write(f"Your progress: {progress_percent}%")
