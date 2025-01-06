import json
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain import hub

#based on https://docs.smith.langchain.com/tutorials/Developers/rag
# Load environment variables from .env file
load_dotenv()

def load_results_from_json(file_path: str) -> pd.DataFrame:
    """Load RAG results from a JSON file into a pandas DataFrame."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

# Grade prompt
answer_accuracy_prompt ='''
SYSTEM:
You are a teacher grading a quiz. 

You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. 
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is not conflicting with the ground truth answer.
Score:
A score of 2 means that the student's answer meets all of the criteria. This is the highest (best) score.
A score of 1 means that the student's answer contains the information from ground truth answer, but not all of it, or its not meating one for the criteria
A score of 0 means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset.
 
'''
grade_prompt_answer_accuracy = hub.pull("langchain-ai/rag-answer-vs-reference")
grade_prompt_answer_accuracy.messages[0].prompt.template = answer_accuracy_prompt
def answer_correctness_evaluator(input_question, reference, prediction, llm, grade_prompt_answer_accuracy=grade_prompt_answer_accuracy) -> dict:
    """
    A simple evaluator for RAG answer accuracy
    """
    # Structured prompt
    answer_grader = grade_prompt_answer_accuracy | llm

    # Run evaluator
    res = answer_grader.invoke({
        "question": input_question,
        "correct_answer": reference,
        "student_answer": prediction
    })
    
    return {"correctness_score": res["Score"], "correctness_explanation": res["Explanation"]}
grade_prompt_answer_helpfulness = prompt = hub.pull("langchain-ai/rag-answer-helpfulness")
def answer_helpfulness_evaluator(input_question, prediction, llm, grade_prompt_answer_helpfulness=grade_prompt_answer_helpfulness) -> dict:
    """
    A simple evaluator for RAG answer accuracy
    """
    # Structured prompt
    answer_grader = grade_prompt_answer_helpfulness | llm

    # Run evaluator
    res = answer_grader.invoke({
        "question": input_question,
        "student_answer": prediction
    })
    
    return {"helpfulness_score": res["Score"], "helpfulness_explanation": res["Explanation"]}

grade_prompt_hallucinations = prompt = hub.pull("langchain-ai/rag-answer-hallucination")
def answer_hallucinations_evaluator(prediction,contexts, llm, grade_prompt_hallucinations=grade_prompt_hallucinations) -> dict:
    """
    A simple evaluator for RAG answer accuracy
    """
    # Structured prompt
    answer_grader = grade_prompt_hallucinations | llm

    # Run evaluator
    res = answer_grader.invoke({"documents": contexts,
                                "student_answer": prediction})
    
    return {"hallucinations_score": res["Score"], "hallucinations_explanation": res["Explanation"]}
grade_prompt_doc_relevance = hub.pull("langchain-ai/rag-document-relevance")
def docs_relevance_evaluator(input_question,contexts, llm, grade_prompt_doc_relevance=grade_prompt_doc_relevance) -> dict:
    """
    A simple evaluator for RAG answer accuracy
    """
    # Structured prompt
    answer_grader = grade_prompt_doc_relevance | llm

    # Run evaluator
    res = answer_grader.invoke({"question":input_question,
                                  "documents":contexts})
    
    return {"relevance_score": res["Score"], "relevance_explanation": res["Explanation"]}

def main():
    input_file = "rag_results_19_9_no_semantic.json"
    output_file = "ragas_evaluation_results_no_semantic.csv"

    # Initialize the Azure OpenAI LLM
    azure_api_key = os.getenv("OPENAI_API_KEY")
    azure_endpoint = os.getenv("OPENAI_DEPLOYMENT_ENDPOINT")
    azure_deployment = os.getenv("OPENAI_DEPLOYMENT_NAME")

    if not all([azure_endpoint, azure_deployment]):
        raise ValueError("One or more Azure OpenAI environment variables are missing")
    
    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        model=azure_deployment,
        temperature=0,
        max_tokens=4000,
        api_version="2023-05-15"
    )

    # Load data into DataFrame
    df = load_results_from_json(input_file)

    # Apply evaluation to each row
    df[['correctness_score', 'correctness_explanation']] = df.apply(lambda row: pd.Series(
        answer_correctness_evaluator(row['question'], row['reference_answer'], row['rag_answer'], llm)
    ), axis=1)
    df[['helpfulness_score', 'helpfulness_explanation']] = df.apply(lambda row: pd.Series(
            answer_helpfulness_evaluator(row['question'], row['rag_answer'], llm)
        ), axis=1)
    df[['hallucinations_score', 'hallucinations_explanation']] = df.apply(lambda row: pd.Series(
            answer_hallucinations_evaluator(contexts=row['contexts'], prediction=row['rag_answer'], llm=llm)
        ), axis=1)
    df[['relevance_chanks_score', 'relevance_chanks_explanation']] = df.apply(lambda row: pd.Series(
            docs_relevance_evaluator(contexts=row['contexts'], input_question=row['question'], llm=llm)
        ), axis=1)
    # Save results to CSV
    df.to_csv(output_file, index=False)
    print(f"Evaluation results saved to {output_file}")

    # Print aggregated results
    correctness_mean_score = df['correctness_score'].mean()
    correctness_mean_score = correctness_mean_score/2
    print(f"Aggregated Results - correctness Mean Score: {correctness_mean_score:.2f}")
    helpfulness_mean_score = df['helpfulness_score'].mean()
    print(f"Aggregated Results - helpfulnesss Mean Score: {helpfulness_mean_score:.2f}")
    hallucinations_mean_score = df['hallucinations_score'].mean()
    print(f"Aggregated Results - hallucinations Mean Score: {hallucinations_mean_score:.2f}")
    relevance_chanks_mean_score = df['relevance_chanks_score'].mean()
    print(f"Aggregated Results - relevance chanks Mean Score: {relevance_chanks_mean_score:.2f}")
if __name__ == "__main__":
    main()
