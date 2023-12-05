import tempfile
import camelot
import pandas as pd
import streamlit as st
from transformers import TapasTokenizer, TapasForQuestionAnswering

# Load Hugging Face token from environment variable
from dotenv import load_dotenv
import os

load_dotenv()
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')

def extract_tables_from_pdf(uploaded_pdf):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
        temp_pdf.write(uploaded_pdf.getbuffer())
        temp_pdf_path = temp_pdf.name

    tables = camelot.read_pdf(temp_pdf_path, pages='all', flavor='lattice')
    return [table.df for table in tables]

def preprocess_dataframe_for_tapas(df):
    # Convert all data to strings and fill NaN values with a placeholder
    df = df.astype(str).fillna("N/A")
    # Ensure column names are strings
    df.columns = df.columns.map(str)
    # Reset the index to ensure the index is also in string format
    df.reset_index(drop=True, inplace=True)
    return df

def answer_question(question, table_df):
    # Preprocess the DataFrame for TAPAS
    table_df = preprocess_dataframe_for_tapas(table_df)
    
    # Load TAPAS tokenizer and model
    tokenizer = TapasTokenizer.from_pretrained("google/tapas-large-finetuned-wtq", use_auth_token=huggingface_token)
    model = TapasForQuestionAnswering.from_pretrained("google/tapas-large-finetuned-wtq", use_auth_token=huggingface_token)

    # Encode the question and the table
    inputs = tokenizer(table=table_df, queries=[question], padding='max_length', return_tensors="pt")
    outputs = model(**inputs)

    # Get the most likely answer tokens
    predicted_answer_coordinates = tokenizer.convert_logits_to_predictions(inputs, outputs.logits.detach())[0]
    
    # We take the first one as the most likely answer (since we have only one question)
    # and extract the corresponding answer from the table.
    if predicted_answer_coordinates:
        # Convert the coordinates to an actual answer
        answer_coordinates = predicted_answer_coordinates[0]
        table_answers = []
        for coordinate in answer_coordinates:
            if coordinate is not None:
                # Convert the predicted answer coordinates to the actual answer in the table
                row_index, column_index = coordinate
                table_answers.append(table_df.iat[row_index, column_index])
        answer = ", ".join(table_answers)  # Join all pieces of the answer
    else:
        answer = "No answer found."

    return answer


def main():
    st.set_page_config(layout="wide")  # Optional: Set the layout to wide
    st.title('PDF Table Extractor and Question Answering System')

    uploaded_pdf = st.file_uploader("Upload a PDF containing tables", type="pdf")
    
    if uploaded_pdf:
        try:
            extracted_tables = extract_tables_from_pdf(uploaded_pdf)
            table_names = [f"Table {i+1}" for i in range(len(extracted_tables))]
            st.success('Tables extracted successfully!')
        except Exception as e:
            st.error(f'An error occurred when extracting tables: {e}')
            return

        # Sidebar for table selection
        st.sidebar.markdown("## Select Table")
        table_to_view = st.sidebar.selectbox("View Table", table_names, index=0)
        view_index = table_names.index(table_to_view)
        st.subheader(f'Viewing {table_to_view}')
        st.dataframe(extracted_tables[view_index])

        # Sidebar for asking question
        st.sidebar.markdown("## Ask a Question")
        table_to_question = st.sidebar.selectbox("Ask Question On", table_names, index=0)
        question_index = table_names.index(table_to_question)
        user_question = st.sidebar.text_input("Enter your question", key="question_input")

        if user_question:
            answer = answer_question(user_question, extracted_tables[question_index])
            st.subheader('Answer')
            st.write(answer)

if __name__ == "__main__":
    main()
