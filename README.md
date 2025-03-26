# 🚀 Project Name

## 📌 Table of Contents
- [Introduction](#introduction)
- [Demo](#demo)
- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [Challenges We Faced](#challenges-we-faced)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## 🎯 Introduction
I've developed an innovative solution that transforms how financial institutions handle regulatory compliance. My system uses Generative AI to automatically extract validation rules from complex regulatory documents and applies them to securities data with precision. By combining rule-based validation with machine learning anomaly detection, my tool identifies both explicit violations and subtle statistical outliers that traditional methods might miss. When issues are found, the system generates specific remediation suggestions, explaining not just what to fix but why it matters for compliance. What makes this approach unique is its adaptability—as regulations evolve, the system learns and updates without manual recoding. The conversational interface I've built makes complex compliance tasks accessible even to users without technical expertise, allowing them to interact naturally with the system and refine rules through dialogue. This solution transforms regulatory reporting from a time-consuming burden into a streamlined, accurate process that helps financial institutions stay ahead of compliance requirements.

## 🎥 Demo
📹 [Video Demo](https://drive.google.com/file/d/1OEbYmvn8c5XKv-VwvFgUkNhLYS77nBov/view?usp=drive_link) (Video starts with a quick presentation then working demo)  
🖼️ [Screenshots](https://drive.google.com/file/d/1-IB7SNBfuAITRo0rWFJe_apkCyCIJh_2/view?usp=sharing) (Please check my artifacts/resultsamples folder for other UI and result screenshots)


## 💡 Inspiration
What struck me most was the disconnect between the intelligence of financial professionals and the rudimentary tools they were using. These experts understand complex regulations intuitively, yet they're forced to translate that understanding into rigid, hand-coded validation rules that can't adapt to new scenarios.
I wanted to create a solution that bridges this gap—leveraging AI to automate the tedious parts of compliance while amplifying human expertise. By combining GenAI's ability to interpret regulatory text with ML's talent for spotting anomalies, my system helps compliance teams focus on what truly matters: ensuring their data tells the right story to regulators.

## ⚙️ What It Does
My regulatory compliance automation system streamlines the entire securities reporting process through the following integrated capabilities:
### 1. **Automated Regulatory Document Processing**  
- Utilizes natural language processing (NLP) through Hugging Face inference API leveraging Mistral-7B-Instruct-v0.2 to extract structured validation requirements from regulatory documents.  
- Eliminates manual effort in interpreting compliance requirements.  

### 2. **Transformation of Rules into Executable Validation Logic**  
- A rule extraction engine converts extracted regulatory requirements into executable validation rules.  
- Removes the need for manual rule definition, ensuring consistency and efficiency.  

### 3. **Automated Data Validation for Compliance Checks**  
- Applies generated validation rules to securities data.  
- Identifies non-compliant records and specific violations efficiently.  

### 4. **Anomaly Detection Using Advanced Machine Learning**  
- Implements an unsupervised anomaly detection system with Isolation Forest algorithms.  
- Flags statistical outliers and unusual patterns that traditional rule-based checks might miss.  

### 5. **Automated Compliance Issue Remediation**  
- A remediation engine generates actionable suggestions for resolving detected compliance issues.  
- Provides explanations of underlying regulatory requirements and recommends data corrections.  

### 6. **User-Friendly Interaction and Accessibility**  
- Features both a structured dashboard interface and a conversational assistant.  
- Enables users to interact with the system using natural language for ease of access.  

### 7. **Continuous Adaptation to Regulatory Changes**  
- Maintains an adaptive rule repository that evolves with regulatory updates and user feedback.  
- Ensures ongoing compliance improvement over time.  

### 8. **Enhanced Efficiency and Accuracy in Compliance Processes**  
- Reduces validation time from days to minutes.  
- Improves accuracy and ensures auditable documentation of the entire compliance process.  

## 🛠️ How We Built It
To build the regulatory compliance automation system, I strategically combined various technologies and methodologies to ensure accuracy, flexibility, and maintainability. The system was designed with a modular architecture, leveraging advanced natural language processing, machine learning, and data processing techniques. Below are the key components of the technical implementation:  

### 1. **Modular Python Framework for Core Architecture**  
- Designed a modular Python framework with distinct components for document processing, rule extraction, data validation, anomaly detection, and remediation generation.  
- Ensured clear separation of concerns, allowing independent testing and scalability.  

### 2. **Natural Language Processing for Regulatory Text Interpretation**  
- Integrated Hugging Face's Inference API with the Mistral-7B-Instruct-v0.2 model.  
- Achieved a balance between performance and accuracy for extracting compliance requirements from regulatory text.  

### 3. **Machine Learning-Based Anomaly Detection**  
- Implemented scikit-learn’s Isolation Forest algorithm to detect statistical anomalies in high-dimensional data.  
- Effectively identified outliers without requiring labeled training examples.  

### 4. **Data Processing and Validation Execution**  
- Utilized pandas for efficient structured data manipulation and preprocessing.  
- Validation logic was implemented as dynamically generated Python code that executes against real-time data.  

### 5. **User-Friendly Interactive Dashboard**  
- Built the user interface using Streamlit for rapid prototyping and interactive visualization.  
- Enabled users to view compliance results and access remediation suggestions through an intuitive dashboard.  

### 6. **Efficient Rule Storage and Persistence Mechanism**  
- Developed a JSON-based rule storage system for maintaining extracted and dynamically generated compliance rules.  
- Ensured flexibility and easy updates to validation rules as regulations evolve.  

### 7. **Automated Testing and Maintainability**  
- Structured the application to support modular testing using Python’s unittest framework.  
- Facilitated maintainability, allowing seamless adaptation to different regulatory frameworks while ensuring consistent performance and accuracy.

## 🚧 Challenges We Faced
Developing this regulatory compliance automation system presented several significant challenges. The most formidable technical hurdle was extracting precise validation rules from regulatory documents, which often contain ambiguous language and implicit requirements. I experimented with multiple prompt engineering approaches to guide the language model toward generating executable validation code rather than merely summarizing regulations.
Another major challenge was handling the wide variety of securities data formats and edge cases. Each validation rule needed to be robust enough to handle missing data, unexpected values, and different data types without generating false positives or negatives. This required developing a sophisticated rule formalization pipeline that could generate context-aware validation code.
The anomaly detection component presented its own challenges, particularly in balancing sensitivity and specificity. Too sensitive, and normal variations would trigger false alarms; too lenient, and genuine anomalies would go undetected. I addressed this through careful parameter tuning and implementing a multi-faceted approach to anomaly explanation.
Integrating the conversational interface with the technical backend proved challenging as well. The system needed to translate natural language queries into specific technical operations and explain complex validation results in accessible terms. This required developing a structured command recognition system with fallback mechanisms for handling ambiguous requests.

## 🏃 How to Run
1. Clone the git hub repo using this command (open a terminal in your file location) - git clone https://github.com/ewfx/gaidp-maayuzz
2. Open the repo folder using a code editor like VSCode and open the terminal in the code editor
3. Create a new branch if you wish
4. Create a python venv (after going inside the "code" directory using this command in terminal  - python -m venv venv
5. Activate the venv using this command - venv\Scripts\activate  
6. pip install everything from the requirements.txt (Please wait for 3 to 4 minutes for it to install)
7. In the .env file present in root directory("code" directory), please paste your Hugging Face API token(Do not paste your token in "", Sample code line(and the only line to be added) in the .env file is : 
HUGGINGFACE_API_KEY=hf_XXXXXXXXXXiR). (To generate a token, go to huggingface.co and create the access token, give read access)
8. To run the streamlit app, please use command - streamlit run app.py
9. Please upload a .txt file in the streamlit Ui for regulatory instructions, I am uploading the sample txt I used in inputs folder in my artifacts.
10. To close the streamlit app, or any process, type combination 'Ctrl + c' in the terminal.
11. To run specific tests, please use command sample - python test\test_rule_extraction.py (The results should mostly appear in terminal)
   ```

## 🏗️ Tech Stack
🔹 Frontend: Streamlit web interface with interactive dashboard
🔹 Backend: Python-based modular architecture
🔹 Machine Learning: scikit-learn for Isolation Forest anomaly detection
🔹 NLP: Hugging Face Inference API with Mistral-7B-Instruct-v0.2
🔹 Data Processing: Pandas for structured data manipulation
🔹 Storage: JSON-based rule repository
🔹 Testing: Python unittest framework for component verification

## 👥 Team
- Team Name - Maayuzz
- Syed Maaiz Syed Shabbeer Basha - [GitHub](#) | [LinkedIn](#)

