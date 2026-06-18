"""
Streamlit app for converting documents into a chatbot using Docling and LangGraph.
"""

import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from streamlit_extras.bottom_container import bottom
import datetime

# Load environment variables
load_dotenv()

# Import our modules
from src.document_processor import DocumentProcessor
from src.vectorstore import VectorStoreManager
from src.tools import create_search_tool, save_feedback
from src.agent import create_documentation_agent
from src.structure_visualizer import DocumentStructureVisualizer

@st.cache_resource(show_spinner="🤖 Taalmodel laden...")
def load_local_agent(_vectorstore):
    return create_documentation_agent(_vectorstore)

# Page configuration
st.set_page_config(
    page_title="Marckus", page_icon="logo.jpg", layout="wide"
)

def initialize_session_state():
    """Initialize all session state variables."""
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = "not_started"
    if "docling_docs" not in st.session_state:
        st.session_state.docling_docs = []


def process_and_index(uploaded_files):
    """Process uploaded documents and create vector store."""
    try:

        load_local_agent.clear()
        
        for key in[
            "summary_result",
            "term_frequency_result",
            "bigrams_result",
        ]:
            if key in st.session_state:
                del st.session_state[key]

        st.session_state.message = []

        # Step 1: Process documents with Docling
        with st.spinner(
            f"📄 {len(uploaded_files)} document(en) verwerken met Docling..."
        ):
            processor = DocumentProcessor()
            documents, docling_docs = processor.process_uploaded_files(uploaded_files)
            st.session_state.docling_docs = docling_docs

        if not documents:
            st.error(
                "Er zijn geen documents verwerkt. Controleer de bestanden en probeer opnieuw."
            )
            return

        # Step 2: Chunk and create vector store
        with st.spinner("✂️ Documenten chunken..."):
            vs_manager = VectorStoreManager()
            chunks = vs_manager.chunk_documents(documents)

        with st.spinner("🔢 Vector store aanmaken..."):
            vectorstore = vs_manager.create_vectorstore(chunks)
            st.session_state.vectorstore = vectorstore

        # Step 3: Create agent
        with st.spinner("🤖 Taalmodel laden..."):
            st.session_state.agent = load_local_agent(vectorstore)

        st.session_state.processing_status = "completed"
        st.success("✅ Documenten geïndexeerd! Je kunt nu hieronder met ze chatten.")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.session_state.processing_status = "error"


def render_sidebar():
    """Render the sidebar with setup controls."""
    with st.sidebar:
        st.title("⚙️ Uploaden")

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Documenten",
            type=["pdf", "docx", "pptx", "html"],
            accept_multiple_files=True,
            help="Upload PDF, Word (DOCX), PowerPoint (PPTX) of HTML bestanden",
        )

        # Show uploaded files count
        if uploaded_files:
            st.info(f"📊 {len(uploaded_files)} bestand(en) geüpload")

            # List uploaded files
            with st.expander("📁 Geüploade bestanden"):
                for file in uploaded_files:
                    st.write(f"- {file.name} ({file.type})")

            # Process button
            if st.button("🚀 Verwerk documenten", use_container_width=True):
                st.session_state.uploaded_files = uploaded_files
                process_and_index(uploaded_files)

        # Status indicator
        st.divider()
        st.subheader("📊 Status")

        if st.session_state.processing_status == "not_started":
            st.info("Klaar om te starten")
        elif st.session_state.processing_status == "completed":
            st.success("✅ Klaar om te chatten!")
        elif st.session_state.processing_status == "error":
            st.error("❌ Er is een fout opgetreden")

        # Tips
        with st.expander("💡 Tips"):
            st.markdown(
                """
            **Geaccepteerde formaten:**
            - PDF documenten (.pdf)
            - Word documenten (.docx)
            - PowerPoint presentaties (.pptx)
            - HTML bestanden

            **Best practices:**
            - Upload gerelateerde documenten samen
            - Start met een paar documenten voor testen
            - Documenten worden verwerkt met OCR voor gescande content
            - Tabellen en structuur worden behouden
            """
            )

        st.divider()
        st.subheader("🤷🏽‍♀️ Voorbeeld vragen")
        st.markdown(
            """
        - Geef een beknopte samenvatting over [onderwerp] op basis van de documenten. 
        - Wordt er een missie omschreven en zo ja, wat is deze missie?
        - Wordt er een visie omschreven en zo ja, wat is deze visie?
        - Wordt er een propositie omschreven en zo ja, wat is deze propositie?
        - Wordt er een positionering omschreven en zo ja, wat is deze positionering?
        - Wordt er een doelgroep omschreven en zo ja, wat is deze doelgroep?
        - Worden er concurrenten genoemd en zo ja, wie zijn deze concurrenten?
            """
        )

def render_structure_viz():
    """Render document structure visualization."""
    st.title("📇 Document Structuur")

    if not st.session_state.docling_docs:
        st.info("👈🏽 Upload eerst je bestanden voordat je de structuur kunt analyseren.")
        return

    # Document selector
    doc_names = [doc['filename'] for doc in st.session_state.docling_docs]
    selected_doc_name = st.selectbox("Selecteer document om te analyseren:", doc_names)

    # Get selected document
    selected_doc_data = next(
        (doc for doc in st.session_state.docling_docs if doc['filename'] == selected_doc_name),
        None
    )

    if not selected_doc_data:
        return

    # Create visualizer
    visualizer = DocumentStructureVisualizer(selected_doc_data['doc'])

    # Display structure in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Veelvoorkomende termen", "Tabellen", "Afbeeldingen", "Check aanwezige onderdelen"])

    with tab1:
        st.subheader("🔤 Veelvoorkomende termen")
        st.markdown("Hier kun je de meest voorkomende woorden en woordcombinaties (bigrams) in het document bekijken. Dit kan helpen om snel inzicht te krijgen in de belangrijkste thema's en onderwerpen die in het document aan bod komen.")

        if st.button("Bereken veelvoorkomende termen"):
            with st.spinner("Term frequency en bigrams berekenen..."):
                term_frequency_df = visualizer.get_term_frequency(
                    top_n=30,
                    min_frequency=3
                )
                bigrams_df = visualizer.get_bigrams(
                    top_n=30,
                    min_frequency=3
                )

                st.session_state["term_frequency_result"] = term_frequency_df
                st.session_state["bigrams_result"] = bigrams_df

        if "term_frequency_result" in st.session_state:
            st.subheader("🔤 Enkele woorden")

            term_frequency_df = st.session_state["term_frequency_result"]

            if not term_frequency_df.empty:
                st.dataframe(term_frequency_df, width="stretch")
            else:
                st.info("Geen termen gevonden die vaak genoeg voorkomen.")

        if "bigrams_result" in st.session_state:
            st.divider()
            st.subheader("🔤🔤 Combinatie woorden")

            bigrams_df = st.session_state["bigrams_result"]

            if not bigrams_df.empty:
                st.dataframe(bigrams_df, width="stretch")
            else:
                st.info("Geen bigrams gevonden die vaak genoeg voorkomen.")
        else:
            st.info("Klik op de knop om termen en bigrams te berekenen.")

    with tab2:
        st.subheader("🔎 Tabellen")
        st.markdown("Hier kun je de tabellen in het document bekijken. Dit kan helpen om snel inzicht te krijgen in de data en informatie die in het document wordt gepresenteerd.")
        tables_info = visualizer.get_tables_info()

        if tables_info:
            for table_data in tables_info:
                st.markdown(f"### Tabel {table_data['table_number']}")

                if table_data['caption']:
                    st.caption(table_data['caption'])

                if not table_data['is_empty']:
                    st.dataframe(table_data['dataframe'], use_container_width=True)
                else:
                    st.info("Tabel is leeg")

                st.divider()
        else:
            st.info("Geen tabellen gevonden in dit document")

    with tab3:
        st.subheader("📸 Afbeeldingen")
        st.markdown("Hier kun je de afbeeldingen in het document bekijken. Dit kan helpen om snel inzicht te krijgen in de visuele elementen die in het document aan bod komen.")
        pictures_info = visualizer.get_pictures_info()

        if pictures_info:
            for pic_data in pictures_info:
                st.markdown(f"**Afbeelding {pic_data['picture_number']}**")

                if pic_data['caption']:
                    st.caption(pic_data['caption'])

                # Display the actual image if available
                if pic_data['pil_image'] is not None:
                    st.image(pic_data['pil_image'], use_container_width=True)
                else:
                    st.info("Afbeeldingsdata niet beschikbaar")

                # Show bounding box info
                if pic_data['bounding_box']:
                    bbox = pic_data['bounding_box']
                    with st.expander("📐 Positiedetails"):
                        st.text(f"Positie: ({bbox['left']:.1f}, {bbox['top']:.1f}) - ({bbox['right']:.1f}, {bbox['bottom']:.1f})")

                st.divider()
        else:
            st.info("Geen afbeeldingen gevonden in dit document")

    with tab4:
        st.subheader("🎈 Check aanwezige onderdelen")
        st.markdown("##### Hier kun je aanvinken welke onderdelen voor je klant aanwezig zijn.")
        st.markdown("Klik aan wat er aanwezig is en houdt zo een overzicht bij van de ontbrekende onderdelen.")
        missie = st.checkbox("Missie 🚀")
        visie = st.checkbox("Visie 🔭")
        propositie = st.checkbox("Propositie 💡")
        positionering = st.checkbox("Positionering 📍")
        doelgroep = st.checkbox("Doelgroep 👥")
        concurrenten = st.checkbox("Concurrenten ⚔️")

        if all([missie, visie, propositie, positionering, doelgroep, concurrenten]):
            st.balloons()
            st.success("🎉 Alle onderdelen zijn aanwezig!")


def render_feedback():
    st.subheader("✍🏽 Gebruikers Feedback")
    st.markdown("Geef hier feedback over de applicatie.")
    with st.expander("🤔 Denk hierbij aan"):
        st.markdown(
                """
            - Gebruiksvriendelijkheid
            - Snelheid van verwerking
            - Nauwkeurigheid van antwoorden
            - Functionaliteiten die je mist
            - Algemeen ontwerp en layout
            """
        )
    st.markdown("Geef hieronder de datum, je rol en je specifieke feedback over de applicatie:")

    data = st.date_input("Datum feedback", value=None)
    rol = st.selectbox("Jouw rol (kies er een)", ["Director", "Projectmanager", "Designer", "Ontwikkelaar", "Zeg ik liever niet", "Anders"])
    onderdeel = st.selectbox("Onderdeel feedback (kies er een)", ["Chat (Marckus)", "Samenvatting", "Veelvoorkomende woorden", "Tabellen", "Afbeeldingen", "Algemeen"])
    score = st.slider("Welk cijfer geef je de applicatie met de huidige functionaliteiten?", 1, 10, 5)
    feedback = st.text_area("Jouw feedback")

    if st.button("📥 Feedback opslaan"):
        if feedback.strip() == "":
            st.warning("⚠️ Feedback veld is leeg. Vul alsjeblieft je feedback in voordat je opslaat.")

        else:
            save_feedback(data, rol, onderdeel, score, feedback)
            st.success("✅ Feedback opgeslagen! Bedankt voor je input.")
        

def render_chat():
    """Render the chat interface."""
    # Check if agent is ready
    if st.session_state.agent is None:
        st.info("👈🏽 Upload eerst je bestanden voordat je met ze kunt chatten.")
        st.markdown(
            """
        ### Hoe te gebruiken:
        1. Upload je documenten in de sidebar (PDF, DOCX, PPTX, or HTML)
        2. Klik op "Verwerk documenten" en wacht op verwerking
        3. Begin met vragen stellen over je documenten!

        ### Wat je kunt doen:
        - Vragen stellen over de inhoud van documenten
        - Informatie vergelijken over verschillende documents
        - Specifieke data of inzichten uit de tekst halen
        - Samenvattingen maken van documentsecties
        """
        )
        return

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input in bottom container (attempt to fix positioning in tabs)
    with bottom():
        prompt = st.chat_input("Stel een vraag over je documenten...")

    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            message_placeholder = st.empty()

            try:
                status_placeholder.markdown("🔍 **Relevante documentfragmenten zoeken...**")

                status_placeholder.markdown("💬 **Antwoord genereren...**")

                full_response = ""
                
                for chunk in st.session_state.agent.invoke(prompt):
                    full_response += chunk
                    message_placeholder.markdown(full_response)

                status_placeholder.empty()
                

            except Exception as e:
                import traceback

                error_details = traceback.format_exc()
                print(f"Chat error: {error_details}")

                error_msg = f"❌ Error: {str(e)}"
                status_placeholder.empty()
                message_placeholder.markdown(error_msg)
                full_response = error_msg

        # Add assistant response to history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )


def main():
    """Main application function."""
    initialize_session_state()
    render_sidebar()

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["🕵🏻‍♀️ Marckus", "📇 Document Structuur", "✍🏽 Gebruikers Feedback"])

    with tab1:
        render_chat()

    with tab2:
        render_structure_viz()

    with tab3:
        render_feedback()


if __name__ == "__main__":
    main()