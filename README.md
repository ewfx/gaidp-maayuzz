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
🔗 [Live Demo](#) (if applicable)  Please check my artifacts/demo folder for a live video and the presentation that was used.
📹 [Video Demo](#) (if applicable)  
🖼️ Screenshots: Please check my artifacts/resultsamples folder for UI and result screenshots

![Screenshot 1](link-to-image)

## 💡 Inspiration
What struck me most was the disconnect between the intelligence of financial professionals and the rudimentary tools they were using. These experts understand complex regulations intuitively, yet they're forced to translate that understanding into rigid, hand-coded validation rules that can't adapt to new scenarios.
I wanted to create a solution that bridges this gap—leveraging AI to automate the tedious parts of compliance while amplifying human expertise. By combining GenAI's ability to interpret regulatory text with ML's talent for spotting anomalies, my system helps compliance teams focus on what truly matters: ensuring their data tells the right story to regulators.

## ⚙️ What It Does
My regulatory compliance automation system streamlines the entire securities reporting process through five integrated capabilities:
First, it processes regulatory documents using natural language processing to extract structured validation requirements. The rule extraction engine then transforms these requirements into executable validation rules, eliminating the need for manual rule definition.
The data validation component applies these rules to securities data, identifying non-compliant records and specific violations. Complementing this rule-based approach, the unsupervised anomaly detection system identifies statistical outliers using Isolation Forest algorithms, flagging unusual patterns that might escape traditional rule checking.
For any compliance issues detected, the remediation engine generates actionable suggestions, explaining the underlying regulatory requirement and recommending specific data corrections. All of these components are accessible through both a structured dashboard interface and a conversational assistant that allows users to interact with the system using natural language.
The solution maintains an adaptive rule repository that evolves based on regulatory changes and user feedback, ensuring continuous compliance improvement over time. This comprehensive approach reduces validation time from days to minutes while significantly improving accuracy and providing auditable documentation of the entire compliance process.

## 🛠️ How We Built It
To create this regulatory compliance automation system, I leveraged a strategic combination of technologies:
For the core architecture, I designed a modular Python framework with separate components for document processing, rule extraction, data validation, anomaly detection, and remediation generation. This approach ensured clear separation of concerns and allowed for independent testing of each component.
The natural language processing capabilities were implemented using Hugging Face's Inference API, specifically employing the Mistral-7B-Instruct-v0.2 model, which provided the right balance of performance and accuracy for regulatory text interpretation.
For anomaly detection, I implemented scikit-learn's Isolation Forest algorithm, which efficiently identifies outliers in high-dimensional data without requiring labeled training examples.
Data processing was handled through pandas for structured data manipulation, while the validation logic was implemented as dynamically generated Python code that executes against the data.
The user interface was built with Streamlit, enabling rapid development of an interactive dashboard that visualizes compliance results and provides remediation suggestions.
For persistence, I implemented a JSON-based rule storage system, and the entire application was structured to enable modular testing using Python's unittest framework.
This technical architecture enables a flexible, maintainable solution that can adapt to different regulatory frameworks while maintaining consistent performance and accuracy.

## 🚧 Challenges We Faced
Developing this regulatory compliance automation system presented several significant challenges. The most formidable technical hurdle was extracting precise validation rules from regulatory documents, which often contain ambiguous language and implicit requirements. I experimented with multiple prompt engineering approaches to guide the language model toward generating executable validation code rather than merely summarizing regulations.
Another major challenge was handling the wide variety of securities data formats and edge cases. Each validation rule needed to be robust enough to handle missing data, unexpected values, and different data types without generating false positives or negatives. This required developing a sophisticated rule formalization pipeline that could generate context-aware validation code.
The anomaly detection component presented its own challenges, particularly in balancing sensitivity and specificity. Too sensitive, and normal variations would trigger false alarms; too lenient, and genuine anomalies would go undetected. I addressed this through careful parameter tuning and implementing a multi-faceted approach to anomaly explanation.
Integrating the conversational interface with the technical backend proved challenging as well. The system needed to translate natural language queries into specific technical operations and explain complex validation results in accessible terms. This required developing a structured command recognition system with fallback mechanisms for handling ambiguous requests.

## 🏃 How to Run
1. Clone the git hub repo using this command (open a terminal in your file location) - git clone https://github.com/ewfx/gaidp-maayuzz
2. Open the repo folder using a code editor like VSCode and open the terminal in the code editor
3. Create a new branch if you wish
4. Create a python venv using this command in terminal  - python -m venv venv
5. Activate the venv using this command - venv\Scripts\activate  
6. pip install everything from the requirements.txt
7. In the .env file present in root directory, please paste your Hugging Face API token(Do not paste your token in "", Sample code line in the .env file is : 
HUGGINGFACE_API_KEY=hf_XXXXXXXXXXiR). (To generate a token, go to huggingface.co and create the access token, give read access)
7. To run the streamlit app, please use command - streamlit run app.py
8. To run specific tests, please use command sample - python tests\test_rule_extraction.py (The results should mostly appear in terminal)
9. To close the streamlit app, or any process, type combination 'Ctrl + c' in the terminal.
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
- **Teammate 2** - [GitHub](#) | [LinkedIn](#) test new
