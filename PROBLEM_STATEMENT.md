# 🧪 RAG & LLM Assessment — Problem Statement

## Background

You are a backend engineer at **HealthBot Inc.**, a health-tech startup building an AI-powered assistant that answers employee questions about the company's **HR policies** (leave policy, insurance, code of conduct, etc.).

The company has a collection of **internal policy documents** (PDFs/text files) that are updated quarterly. The assistant must always provide answers grounded in these documents — **not from the LLM's general knowledge** — to avoid hallucinations and ensure compliance.

---

## Problem Statement

### Build a RAG-based Q&A System for Internal HR Policy Documents

Design and implement a **Retrieval-Augmented Generation (RAG)** pipeline that:

1. **Ingests** a set of HR policy documents (provided as `.txt` files).
2. **Chunks** the documents into meaningful segments.
3. **Embeds** the chunks using an embedding model and stores them in a **vector database**.
4. **Retrieves** the most relevant chunks for a given user query.
5. **Generates** a grounded answer using an LLM, citing the source document(s).

---

## Sample Documents

You are provided with **3 sample policy documents** (see the `policy_documents/` directory):

### Document 1: `leave_policy.txt`
```
HealthBot Inc. Leave Policy (Effective Jan 2026)

1. Annual Leave: All full-time employees are entitled to 24 days of paid annual leave per calendar year. Leave is accrued at 2 days per month. Unused leave can be carried forward up to a maximum of 10 days to the next year. Any leave beyond 10 days will lapse on December 31st.

2. Sick Leave: Employees are entitled to 12 days of paid sick leave per year. A medical certificate is required for sick leave exceeding 3 consecutive days. Sick leave cannot be carried forward or encashed.

3. Maternity Leave: Female employees are entitled to 26 weeks of paid maternity leave for the first two children. For the third child onwards, the entitlement is 12 weeks. Maternity leave can be availed up to 8 weeks before the expected delivery date.

4. Paternity Leave: Male employees are entitled to 10 working days of paid paternity leave, to be availed within 6 months of the child's birth.

5. Bereavement Leave: In case of death of an immediate family member (spouse, child, parent, sibling), employees are granted 5 days of paid bereavement leave.

6. Leave Without Pay (LWP): Employees who have exhausted their paid leave quota may apply for LWP with prior approval from their manager and HR. LWP exceeding 30 days in a year may affect annual appraisals and benefits.
```

### Document 2: `insurance_policy.txt`
```
HealthBot Inc. Group Health Insurance Policy (2026)

1. Coverage: All full-time employees and their dependents (spouse and up to 2 children) are covered under the group health insurance plan from the date of joining.

2. Sum Insured: The base sum insured is INR 5,00,000 per family per year. Employees can opt for a top-up plan of INR 10,00,000 or INR 25,00,000 at a subsidized premium.

3. Pre-existing Conditions: Pre-existing conditions are covered after a waiting period of 2 years from the date of enrollment. Diabetes and hypertension are covered from day one with a sub-limit of INR 50,000.

4. Maternity Coverage: Maternity expenses are covered up to INR 75,000 for normal delivery and INR 1,00,000 for C-section, after a waiting period of 9 months from the policy start date.

5. Dental and Vision: Dental treatments are covered up to INR 15,000 per year. Vision correction (spectacles/lenses) is covered up to INR 5,000 per year. LASIK surgery is not covered.

6. Claims Process: Cashless treatment is available at 4000+ network hospitals. For reimbursement claims, submit bills within 15 days of discharge. Claims are processed within 30 working days.

7. Exclusions: Cosmetic surgery, experimental treatments, injuries from adventure sports, and self-inflicted injuries are not covered.
```

### Document 3: `code_of_conduct.txt`
```
HealthBot Inc. Code of Conduct (v3.1)

1. Workplace Behavior: All employees are expected to maintain professional behavior. Harassment, bullying, discrimination based on gender, race, religion, or sexual orientation will not be tolerated and may result in immediate termination.

2. Conflict of Interest: Employees must disclose any personal or financial interest that may conflict with the company's interests. Moonlighting or freelancing for competitors is strictly prohibited without written approval from the CTO and HR.

3. Data Privacy: Employees handling customer or patient data must comply with HIPAA and India's DPDP Act 2023. Unauthorized access, sharing, or downloading of sensitive data is a terminable offense.

4. Intellectual Property: All work products, code, designs, and inventions created during employment belong to HealthBot Inc. Employees must sign an IP assignment agreement within the first week of joining.

5. Social Media Policy: Employees may use social media but must not share confidential company information, client details, or disparaging remarks about the company. Violations may lead to disciplinary action.

6. Reporting Violations: Employees can report violations through the anonymous Ethics Hotline (ethics@healthbot.com) or directly to the Compliance Officer. Retaliation against whistleblowers is strictly prohibited and is a terminable offense.

7. Disciplinary Process: Violations are handled through a 3-step process: (a) Written Warning, (b) Suspension without pay (up to 30 days), (c) Termination. Severe violations (data breach, harassment) may result in immediate termination bypassing steps (a) and (b).
```

---

## Requirements

### Functional Requirements

| # | Requirement | Details |
|---|------------|---------|
| 1 | **Document Ingestion** | Load all `.txt` files from a given directory |
| 2 | **Chunking** | Split documents into chunks (choose an appropriate strategy and chunk size) |
| 3 | **Embedding & Storage** | Generate embeddings and store in a vector DB (e.g., ChromaDB, FAISS, Pinecone) |
| 4 | **Retrieval** | Given a user query, retrieve the top-K most relevant chunks |
| 5 | **Answer Generation** | Use an LLM (OpenAI GPT / open-source) to generate an answer grounded in retrieved context |
| 6 | **Source Citation** | The response must include which document(s) the answer was derived from |

### Non-Functional Requirements

- The system should be modular (separate ingestion, retrieval, and generation stages).
- Handle edge cases: query with no relevant context, ambiguous queries.
- Include a simple way to test (CLI or script with sample queries).

---

## Sample Queries to Test

Your system should be able to answer these correctly:

1. *"How many days of paternity leave can I take?"*
2. *"Is LASIK surgery covered under the insurance plan?"*
3. *"What happens if I moonlight for a competitor?"*
4. *"Can I carry forward my unused sick leave?"*
5. *"What is the sum insured under the health insurance plan?"*
6. *"How do I report an ethics violation?"*

---

## Deliverables

1. **Source code** — Well-structured Python code implementing the RAG pipeline.
2. **README** — Instructions on how to set up and run the system.
3. **Sample output** — Provide outputs for at least 3 of the sample queries above.
4. **Short write-up (200-300 words)** — Explain:
   - Your chunking strategy and why you chose it.
   - Which embedding model you used and why.
   - Which vector DB you chose and why.
   - How you handle queries with no relevant context.
   - Any improvements you would make with more time.

---

## Evaluation Criteria

| Criteria | Weight | What We're Looking For |
|----------|--------|----------------------|
| **Correctness** | 25% | Answers are accurate and grounded in the documents |
| **Architecture** | 20% | Clean separation of concerns, modular design |
| **Chunking Strategy** | 15% | Thoughtful chunk size, overlap, and splitting logic |
| **Retrieval Quality** | 15% | Relevant chunks are retrieved; irrelevant ones are filtered |
| **Edge Case Handling** | 10% | Graceful handling of out-of-scope or ambiguous queries |
| **Code Quality** | 10% | Clean, readable, well-documented code |
| **Write-up Quality** | 5% | Clear articulation of design decisions |

---

## Constraints

- **Time**: 3 hours
- **Language**: Python 3.10+
- **LLM**: You may use OpenAI API (key will be provided) or any open-source model via Ollama/HuggingFace
- **Vector DB**: Your choice (ChromaDB recommended for simplicity)
- You may use LangChain, LlamaIndex, or build from scratch — justify your choice

---

> **⚠️ Important:** The LLM must ONLY use the retrieved document context to answer. Answers from the LLM's general training data are considered failures.

Good luck! 🚀
