## 🏃 How to Run
1. Clone the git hub repo using this command (open a terminal in your file location) - git clone https://github.com/ewfx/gaidp-maayuzz
2. Open the repo folder using a code editor like VSCode and open the terminal in the code editor
3. Create a new branch if you wish
4. Create a python venv (after going inside the "code" directory using this command in terminal  - python -m venv venv
5. Activate the venv using this command - venv\Scripts\activate  
6. pip install everything from the requirements.txt (Please wait for 3 to 4 minutes for it to install)
7. In the .env file present/create a .env file in root directory("code" directory), please paste your Hugging Face API token(Do not paste your token in "", Sample code line(and the only code line to be added) in the .env file is : 
HUGGINGFACE_API_KEY=hf_XXXXXXXXXXiR). (To generate a token, go to huggingface.co and create the access token, give read access)
8. To run the streamlit app, please use command - streamlit run app.py
9. Please upload a .txt file in the streamlit Ui for regulatory instructions, I am uploading the sample txt I used in inputs folder in my artifacts.
10. To close the streamlit app, or any process, type combination 'Ctrl + c' in the terminal.
11. To run specific tests, please use command sample - python test\test_rule_extraction.py (The results should mostly appear in terminal)
