# OpenSpec AGENTS Workflow

This document describes how I (the student) collaborate with AI coding assistants (e.g., ChatGPT, Copilot CLI) using the OpenSpec workflow throughout this project.

---

## 1. Roles

### **Human Developer (Student)**
- Owns project goals, milestones, and final decisions.
- Writes, edits, and reviews specifications (`project.md`, proposals).
- Runs experiments, validates results, fixes environment issues.

### **AI Coding Assistant (ChatGPT / Copilot CLI)**
- Helps draft and refine project specs.
- Suggests implementation details, debugging steps, and code improvements.
- Generates boilerplate code for notebooks, scripts, and Streamlit UI.

### **Tooling**
- Python 3.x environment  
- Jupyter Notebooks (`notebooks/`)  
- Streamlit for interactive demo  
- Git + GitHub for version control  
- scikit-learn, pandas, numpy, matplotlib, seaborn, joblib  

---

## 2. Workflow Overview (How OpenSpec is used)

### **Step 1 — Clarify Requirements**
- Read the homework description.
- Summarize scope, goals, and constraints in `openspec/project.md`.

### **Step 2 — Create Change Proposals**
- For each feature or development milestone:
  - Create a proposal in `openspec/proposals/`.
  - Example:  
    - `001-data-preprocessing.md`  
    - `002-model-training.md`  
    - `003-evaluation.md`  
    - `004-streamlit-ui.md`

- Each proposal contains:
  - Motivation  
  - Requirements  
  - Out of scope  
  - Acceptance criteria  
  - Planned implementation steps  

### **Step 3 — Plan with AI Assistance**
- Ask the AI to review proposals and refine:
  - Implementation details
  - API usage
  - Recommended libraries
  - Edge cases & validation steps

### **Step 4 — Implement the Feature**
- Work in `notebooks/` to implement:
  - Preprocessing
  - Vectorization
  - Model training
  - Evaluation
- Save artifacts into `models/` (e.g., TF-IDF vectorizer, trained model).
- Use AI for debugging or improving code quality.

### **Step 5 — Document & Review**
- Update proposal statuses (Planned → In Progress → Done).
- Update the README with instructions for:
  - Running notebooks  
  - Installing dependencies  
  - Launching Streamlit demo  

### **Step 6 — Deliverables**
- Public GitHub repository  
- Complete OpenSpec documents  
- Functional Streamlit demo  
- Trained model artifacts  
- Evaluation figures and metrics  

---

## 3. Collaboration Rules

### Human Developer Responsibilities
- Validate correctness of AI-generated code.
- Verify that preprocessing and model outputs make sense.
- Ensure no harmful or incorrect assumptions are applied.
- Maintain version control via Git commits.

### AI Assistant Responsibilities
- Provide examples, code, and conceptual explanations.
- Suggest refactors and improvements.
- Assist in documentation and proposal writing.
- Help create visualization code and Streamlit UI components.

---

## 4. Workflow Trace (History Log)

This section records how the OpenSpec workflow has been used so far.

- **Step 1** — Initialized the GitHub repository and created `project.md`.  
- **Step 2** — Created Proposal 001 for data preprocessing.  
- **Step 3** — Used AI assistance to refine core pipeline steps and vectorization plan.  
- **Step 4** — Prepared for model training and evaluation notebook drafts.  
- **Step 5** — Used AI to design Streamlit interface layout and inference workflow.

(Additional entries will be appended as development continues.)

