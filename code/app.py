import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from io import StringIO
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import our components
from src.document_processing import HuggingFaceProcessor
from src.rule_extraction import RuleExtractor
from src.rule_storage import RuleStorage
from src.rule_validation import RuleValidator
from src.anomaly_detection import AnomalyDetector
from src.remediation import RemediationGenerator
from src.conversation import ConversationHandler

# Set page config
st.set_page_config(
    page_title="Regulatory Compliance Assistant",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'conversation_handler' not in st.session_state:
    st.session_state.conversation_handler = ConversationHandler()

if 'document_processor' not in st.session_state:
    st.session_state.document_processor = HuggingFaceProcessor()

if 'rule_extractor' not in st.session_state:
    st.session_state.rule_extractor = RuleExtractor()

if 'rule_storage' not in st.session_state:
    # Create rules directory if it doesn't exist
    os.makedirs("rules", exist_ok=True)
    st.session_state.rule_storage = RuleStorage(storage_dir="rules")

if 'rule_validator' not in st.session_state:
    st.session_state.rule_validator = RuleValidator()

if 'anomaly_detector' not in st.session_state:
    st.session_state.anomaly_detector = AnomalyDetector()

if 'remediation_generator' not in st.session_state:
    st.session_state.remediation_generator = RemediationGenerator()

if 'document_text' not in st.session_state:
    st.session_state.document_text = None

if 'extracted_rules' not in st.session_state:
    st.session_state.extracted_rules = []

if 'data' not in st.session_state:
    st.session_state.data = None

if 'validation_results' not in st.session_state:
    st.session_state.validation_results = None

if 'anomaly_results' not in st.session_state:
    st.session_state.anomaly_results = None

if 'anomaly_details' not in st.session_state:
    st.session_state.anomaly_details = {}

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Define helper functions
def extract_rules_from_document():
    """Extract rules from the regulatory document."""
    if st.session_state.document_text:
        with st.spinner("Extracting rules from document..."):
            # Process the document
            processed_doc = st.session_state.document_processor.process_text(st.session_state.document_text)
            
            # Extract rules
            rules = st.session_state.rule_extractor.extract_rules(processed_doc)
            
            # Save rules to storage
            rules_file = st.session_state.rule_storage.save_rules(rules, "Regulatory Rules")
            
            # Update session state
            st.session_state.extracted_rules = rules
            
            # Update conversation handler context
            st.session_state.conversation_handler.current_document = "Regulatory Document"
            st.session_state.conversation_handler.current_rules = rules
            
            st.success(f"Extracted {len(rules)} rules from the document!")
            return rules
    else:
        st.error("Please upload a regulatory document first.")
        return []

def validate_data():
    """Validate data against extracted rules."""
    if st.session_state.data is None:
        st.error("Please upload data first.")
        return None
    
    if not st.session_state.extracted_rules:
        st.error("Please extract rules from a regulatory document first.")
        return None
    
    with st.spinner("Validating data against rules..."):
        # Create a validator with the extracted rules
        validator = RuleValidator(st.session_state.extracted_rules)
        
        # Validate the data
        validation_results = validator.validate(st.session_state.data)
        
        # Update session state
        st.session_state.validation_results = validation_results
        
        # Update conversation handler context
        st.session_state.conversation_handler.validation_results = validation_results
        
        # Generate validation report
        report = validator.generate_validation_report(validation_results)
        
        st.success("Validation complete!")
        return validation_results, report

def detect_anomalies():
    """Detect anomalies in the data."""
    if st.session_state.data is None:
        st.error("Please upload data first.")
        return None
    
    with st.spinner("Detecting anomalies in data..."):
        # Fit the anomaly detector
        st.session_state.anomaly_detector.fit(st.session_state.data)
        
        # Detect anomalies
        anomaly_results = st.session_state.anomaly_detector.detect_anomalies(st.session_state.data)
        
        # Get anomaly indices
        anomaly_indices = anomaly_results[anomaly_results['anomaly']].index.tolist()
        
        # Get anomaly details
        anomaly_details = st.session_state.anomaly_detector.identify_anomaly_features(st.session_state.data, anomaly_indices)
        
        # Update session state
        st.session_state.anomaly_results = anomaly_results
        st.session_state.anomaly_details = anomaly_details
        
        # Update conversation handler context
        st.session_state.conversation_handler.anomaly_results = anomaly_results
        
        st.success(f"Detected {len(anomaly_indices)} anomalies in the data!")
        return anomaly_results, anomaly_details

def suggest_remediation():
    """Suggest remediation actions for validation issues and anomalies."""
    remediation_results = None
    
    if st.session_state.validation_results is not None:
        with st.spinner("Generating remediation suggestions for validation issues..."):
            # Generate remediation suggestions for validation issues
            remediation_results = st.session_state.remediation_generator.suggest_remediation(
                st.session_state.validation_results, 
                st.session_state.extracted_rules
            )
    
    if st.session_state.anomaly_results is not None:
        with st.spinner("Generating remediation suggestions for anomalies..."):
            # Generate remediation suggestions for anomalies
            if remediation_results is None:
                remediation_results = st.session_state.remediation_generator.suggest_anomaly_remediation(
                    st.session_state.anomaly_results,
                    st.session_state.anomaly_details
                )
            else:
                # Merge remediation suggestions
                anomaly_remediation = st.session_state.remediation_generator.suggest_anomaly_remediation(
                    st.session_state.anomaly_results,
                    st.session_state.anomaly_details
                )
                
                # For each record with anomaly remediation suggestions, add them to the existing suggestions
                for idx, row in anomaly_remediation.iterrows():
                    if row['remediation_suggestions']:
                        if idx in remediation_results.index:
                            remediation_results.at[idx, 'remediation_suggestions'].extend(row['remediation_suggestions'])
                        else:
                            remediation_results.at[idx, 'remediation_suggestions'] = row['remediation_suggestions']
    
    if remediation_results is None:
        st.error("Please validate data or detect anomalies first.")
        return None
    
    st.success("Generated remediation suggestions!")
    return remediation_results

def process_chat_input():
    """Process chat input and update chat history."""
    user_message = st.session_state.chat_input
    
    if user_message:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_message})
        
        # Process the message
        response = st.session_state.conversation_handler.process_message(user_message)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Clear the input
        st.session_state.chat_input = ""

# Create the Streamlit interface
st.title("Regulatory Compliance Assistant")

# Create tabs for different functionalities
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Document Processing", 
    "Rule Extraction", 
    "Data Validation", 
    "Anomaly Detection", 
    "Remediation",
    "Conversation"
])

# Document Processing Tab
with tab1:
    st.header("Document Processing")
    
    # Document upload section
    st.subheader("Upload Regulatory Document")
    uploaded_file = st.file_uploader("Choose a regulatory document", type=["txt"])
    
    if uploaded_file is not None:
        # Read the file content
        if uploaded_file.type == "application/pdf":
            st.warning("PDF parsing is simulated in this demo. Only text content will be processed.")
            # In a real implementation, you would use a PDF parser here
            # For this demo, assume the PDF contains only text
            document_text = uploaded_file.read().decode("utf-8", errors="ignore")
        else:
            document_text = uploaded_file.read().decode("utf-8")
        
        # Save document text to session state
        st.session_state.document_text = document_text
        
        # Show document preview
        st.subheader("Document Preview")
        with st.expander("View Document Content"):
            st.text_area("Document Content", document_text, height=300)
        
        # Update conversation handler context
        st.session_state.conversation_handler.current_document = uploaded_file.name

# Rule Extraction Tab
with tab2:
    st.header("Rule Extraction")
    
    if st.session_state.document_text:
        if st.button("Extract Rules"):
            extract_rules_from_document()
    else:
        st.info("Please upload a regulatory document in the Document Processing tab first.")
    
    # Display extracted rules
    if st.session_state.extracted_rules:
        st.subheader(f"Extracted Rules ({len(st.session_state.extracted_rules)})")
        
        for rule in st.session_state.extracted_rules:
            with st.expander(f"{rule.get('id', 'Unknown')} - {rule.get('name', 'Unnamed Rule')}"):
                st.write(f"**Type:** {rule.get('type', 'Unknown')}")
                st.write(f"**Fields:** {', '.join(rule.get('fields', []))}")
                st.write(f"**Description:** {rule.get('description', 'No description')}")
                
                # Show validation code
                if 'validation_code' in rule:
                    st.code(rule['validation_code'], language="python")

# Data Validation Tab
with tab3:
    st.header("Data Validation")
    
    # Data upload section
    st.subheader("Upload Data")
    data_file = st.file_uploader("Choose a data file", type=["csv", "xlsx"])
    
    if data_file is not None:
        # Read the data file
        try:
            if data_file.name.endswith('.csv'):
                data = pd.read_csv(data_file)
            else:
                data = pd.read_excel(data_file)
            
            # Save data to session state
            st.session_state.data = data
            
            # Update conversation handler context
            st.session_state.conversation_handler.current_data = data
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(data.head())
            
            st.text(f"Loaded {len(data)} records with {len(data.columns)} columns.")
        except Exception as e:
            st.error(f"Error loading data file: {e}")
    
    # Data validation section
    if st.session_state.data is not None and st.session_state.extracted_rules:
        if st.button("Validate Data"):
            validation_results, report = validate_data()
            
            if validation_results is not None:
                # Show validation summary
                st.subheader("Validation Summary")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", report['total_records'])
                with col2:
                    st.metric("Passed", report['passed_records'])
                with col3:
                    st.metric("Failed", report['failed_records'])
                
                # Show pass rate as a progress bar
                st.progress(report['pass_rate'])
                st.text(f"Pass Rate: {report['pass_rate']:.2%}")
                
                # Show rule failure frequency
                st.subheader("Rule Failure Frequency")
                rule_freq_df = pd.DataFrame({
                    'Rule': list(report['rule_frequency'].keys()),
                    'Failures': list(report['rule_frequency'].values())
                })
                
                if not rule_freq_df.empty:
                    st.bar_chart(rule_freq_df.set_index('Rule'))
                else:
                    st.info("No rule failures detected.")
                
                # Show failed records
                st.subheader("Failed Records")
                
                failed_records = validation_results[~validation_results['validation_passed']]
                if not failed_records.empty:
                    st.dataframe(failed_records.drop(['failed_rules', 'validation_messages'], axis=1))
                    
                    for idx, record in failed_records.iterrows():
                        with st.expander(f"Record {idx} - {record.get('Unique_ID', 'Unknown')}"):
                            st.write("**Failed Rules:**")
                            for rule_id in record['failed_rules']:
                                st.write(f"- {rule_id}")
                            
                            st.write("**Validation Messages:**")
                            for message in record['validation_messages']:
                                st.write(f"- {message}")
                else:
                    st.success("All records passed validation!")
    else:
        if st.session_state.data is None:
            st.info("Please upload data first.")
        elif not st.session_state.extracted_rules:
            st.info("Please extract rules from a regulatory document first.")

# Anomaly Detection Tab
with tab4:
    st.header("Anomaly Detection")
    
    if st.session_state.data is not None:
        # Contamination selector (percentage of data expected to be anomalous)
        contamination = st.slider("Expected Anomaly Rate (%)", 1, 20, 5) / 100
        
        if st.button("Detect Anomalies"):
            # Update anomaly detector contamination
            st.session_state.anomaly_detector.contamination = contamination
            
            # Detect anomalies
            anomaly_results, anomaly_details = detect_anomalies()
            
            if anomaly_results is not None:
                # Show anomaly summary
                st.subheader("Anomaly Detection Summary")
                
                anomaly_count = anomaly_results['anomaly'].sum()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Records", len(anomaly_results))
                with col2:
                    st.metric("Anomalies Detected", int(anomaly_count))
                
                # Plot anomaly distribution
                st.subheader("Anomaly Score Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data=anomaly_results, x='anomaly_score', hue='anomaly', kde=True, ax=ax)
                ax.set_title('Distribution of Anomaly Scores')
                ax.set_xlabel('Anomaly Score')
                ax.set_ylabel('Count')
                st.pyplot(fig)
                
                # Show anomalies
                st.subheader("Detected Anomalies")
                
                anomalous_records = anomaly_results[anomaly_results['anomaly']]
                if not anomalous_records.empty:
                    st.dataframe(anomalous_records.drop(['anomaly', 'anomaly_score'], axis=1))
                    
                    for idx, record in anomalous_records.iterrows():
                        with st.expander(f"Anomaly {idx} - {record.get('Unique_ID', 'Unknown')}"):
                            st.write(f"**Anomaly Score:** {record['anomaly_score']:.4f}")
                            
                            # Show anomalous features
                            details = anomaly_details.get(idx, [])
                            if details:
                                st.write("**Anomalous Features:**")
                                for detail in details:
                                    if detail['type'] == 'numeric':
                                        st.write(f"- {detail['feature']}: {detail['value']:.2f} ({detail['reason']})")
                                    else:
                                        st.write(f"- {detail['feature']}: {detail['value']} ({detail['reason']})")
                            else:
                                st.write("No specific anomalous features identified.")
                else:
                    st.success("No anomalies detected!")
    else:
        st.info("Please upload data first.")

# Remediation Tab
with tab5:
    st.header("Remediation Suggestions")
    
    if st.session_state.validation_results is not None or st.session_state.anomaly_results is not None:
        if st.button("Generate Remediation Suggestions"):
            remediation_results = suggest_remediation()
            
            if remediation_results is not None:
                # Show records with remediation suggestions
                st.subheader("Remediation Suggestions")
                
                # Get records with suggestions
                records_with_suggestions = remediation_results[remediation_results['remediation_suggestions'].apply(lambda x: len(x) > 0)]
                
                if not records_with_suggestions.empty:
                    # Display a simplified view of the records
                    st.dataframe(records_with_suggestions.drop(['failed_rules', 'validation_messages', 'remediation_suggestions'], axis=1))
                    
                    # Show detailed suggestions for each record
                    for idx, record in records_with_suggestions.iterrows():
                        with st.expander(f"Record {idx} - {record.get('Unique_ID', 'Unknown')}"):
                            for suggestion in record['remediation_suggestions']:
                                if 'rule_id' in suggestion:
                                    st.write(f"**Suggestion for rule {suggestion['rule_id']}:**")
                                elif 'type' in suggestion and suggestion['type'] == 'anomaly':
                                    st.write(f"**Anomaly remediation suggestion:**")
                                else:
                                    st.write("**Remediation suggestion:**")
                                
                                if 'fields_to_update' in suggestion and suggestion['fields_to_update']:
                                    st.write("Fields to update:")
                                    for field, value in suggestion['fields_to_update'].items():
                                        st.write(f"- {field}: Current value = {value}")
                                
                                if 'suggested_values' in suggestion and suggestion['suggested_values']:
                                    st.write("Suggested values:")
                                    for field, value in suggestion['suggested_values'].items():
                                        st.write(f"- {field}: Suggested value = {value}")
                                
                                if 'explanation' in suggestion:
                                    st.write("Explanation:")
                                    st.write(suggestion['explanation'])
                else:
                    st.info("No remediation suggestions generated.")
                
                # Option to export remediation suggestions
                st.subheader("Export Remediation Suggestions")
                
                if st.button("Export as CSV"):
                    # Prepare data for export
                    export_data = []
                    
                    for idx, record in records_with_suggestions.iterrows():
                        for suggestion in record['remediation_suggestions']:
                            # Get basic record info
                            record_info = {
                                'record_id': idx,
                                'unique_id': record.get('Unique_ID', ''),
                                'suggestion_type': suggestion.get('type', 'validation')
                            }
                            
                            # Add rule info if available
                            if 'rule_id' in suggestion:
                                record_info['rule_id'] = suggestion['rule_id']
                                record_info['rule_name'] = suggestion.get('rule_name', '')
                            
                            # Add fields to update
                            if 'fields_to_update' in suggestion:
                                for field, value in suggestion['fields_to_update'].items():
                                    record_info['field'] = field
                                    record_info['current_value'] = value
                                    
                                    # Add suggested value if available
                                    if 'suggested_values' in suggestion and field in suggestion['suggested_values']:
                                        record_info['suggested_value'] = suggestion['suggested_values'][field]
                                    else:
                                        record_info['suggested_value'] = ''
                                    
                                    # Add explanation
                                    record_info['explanation'] = suggestion.get('explanation', '')
                                    
                                    # Add to export data
                                    export_data.append(record_info.copy())
                    
                    # Convert to DataFrame
                    export_df = pd.DataFrame(export_data)
                    
                    # Convert to CSV and provide download link
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"remediation_suggestions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    else:
        st.info("Please validate data or detect anomalies first.")

# Conversation Tab
with tab6:
    st.header("Conversation Interface")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Assistant:** {message['content']}")
    
    # Chat input
    st.text_input("Ask a question or give a command", key="chat_input", on_change=process_chat_input)
    
    # Provide some example prompts
    with st.expander("Example prompts"):
        st.markdown("""
        - "Extract rules from the regulatory document"
        - "Validate my data against the rules"
        - "Detect anomalies in my data"
        - "Suggest remediation for the issues"
        - "Explain rule rule_1_Unique_ID to me"
        - "Show me a summary of the validation results"
        - "What are the most common validation failures?"
        - "How can I fix the issues in record ID001?"
        """)

# Add a footer
st.markdown("---")
st.markdown("Regulatory Compliance Assistant | Developed using GenAI and ML techniques - By Syed Maaiz")