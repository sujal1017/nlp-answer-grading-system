import re
from typing import Dict, List

def clean_page_markers(markdown_text: str) -> str:
    cleaned = re.sub(r'###\s*Page\s*\d+\s*', '\n', markdown_text, flags=re.IGNORECASE)
    cleaned = re.sub(r'\n---\n', '\n', cleaned)
    return cleaned.strip()

def parse_sdmcet_question_paper(qp_markdown: str) -> List[Dict]:
    """
    Parses hierarchical college Question Papers like SDMCET IA papers:
    Handles 'Q. No. 1' followed by 'a) ... [5M, CO:1]' and 'b) ... [5M, CO:2]'
    """
    text = clean_page_markers(qp_markdown)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    main_q_pattern = re.compile(r'^(?:#+\s*)?Q(?:uestion)?\.?\s*(?:No\.?)?\s*(\d+)\b', re.IGNORECASE)
    sub_q_pattern = re.compile(r'^([a-dA-D])\s*[\)\.]\s*(.+)', re.IGNORECASE)
    marks_co_pattern = re.compile(r'\[\s*(\d+)\s*M(?:arks?)?\s*(?:,\s*CO\s*:\s*(\d+))?\s*\]', re.IGNORECASE)
    
    current_main_q = "1"
    extracted_questions = []
    current_sub_q = None
    
    for line in lines:
        # Skip header metadata lines (USN, Course Code, Note, etc.)
        if any(skip in line.upper() for skip in ["USN:", "COURSE CODE", "INTERNAL ASSESSMENT", "NOTE:", "*****"]):
            continue
            
        main_match = main_q_pattern.match(line)
        if main_match:
            if current_sub_q:
                extracted_questions.append(current_sub_q)
                current_sub_q = None
            current_main_q = main_match.group(1)
            continue
            
        sub_match = sub_q_pattern.match(line)
        if sub_match:
            if current_sub_q:
                extracted_questions.append(current_sub_q)
            sub_letter = sub_match.group(1).upper()
            q_text = sub_match.group(2).strip()
            current_sub_q = {
                "question_id": f"Q{current_main_q}{sub_letter}",
                "main_q": f"Q{current_main_q}",
                "question_text": q_text,
                "max_marks": 5,
                "co": "CO:1"
            }
        elif current_sub_q:
            # Multi-line question text continuation
            current_sub_q["question_text"] += " " + line

        # Check if the current accumulated text has [5M, CO:1]
        if current_sub_q:
            m_match = marks_co_pattern.search(current_sub_q["question_text"])
            if m_match:
                current_sub_q["max_marks"] = int(m_match.group(1))
                if m_match.group(2):
                    current_sub_q["co"] = f"CO:{m_match.group(2)}"
                # Clean the [5M, CO:1] tag out of the question text
                current_sub_q["question_text"] = marks_co_pattern.sub('', current_sub_q["question_text"]).strip()

    if current_sub_q:
        extracted_questions.append(current_sub_q)
        
    return extracted_questions

def chunk_student_booklet(raw_markdown: str, expected_q_ids: List[str] = None) -> Dict[str, str]:
    """
    Splits student booklet into Q1A, Q1B, Q2A, etc.
    Handles '1a)', 'Q1 a)', 'Ans 1a:', or 'Q.No.1 a)'
    """
    text = clean_page_markers(raw_markdown)
    
    pattern = re.compile(
        r'(?:^|\n)\s*(?:#+\s*)?(?:Q(?:uestion)?\.?\s*(?:No\.?)?\s*|Ans(?:wer)?\.?\s*)?(\d+)\s*[\.\)\-]?\s*([a-dA-D])?\s*[\.\)\:\-]\s*',
        re.IGNORECASE
    )
    
    matches = list(pattern.finditer(text))
    if not matches:
        default_qid = expected_q_ids[0].upper() if expected_q_ids else "Q1A"
        return {default_qid: text}
        
    chunks: Dict[str, str] = {}
    for idx, match in enumerate(matches):
        q_num = match.group(1)
        sub_let = (match.group(2) or "").upper()
        q_id = f"Q{q_num}{sub_let}"
        
        start_pos = match.end()
        end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = text[start_pos:end_pos].strip()
        
        if q_id in chunks:
            chunks[q_id] += f"\n\n{content}"
        else:
            chunks[q_id] = content
            
    return chunks

def calculate_ia_total(scores_by_qid: Dict[str, float]) -> float:
    """
    Applies SDMCET IA choice rule:
    'Answer any one full question from Q.No.1 & Q.No.2; Q.No.3 is COMPULSORY.'
    """
    q1_total = scores_by_qid.get("Q1A", 0.0) + scores_by_qid.get("Q1B", 0.0)
    q2_total = scores_by_qid.get("Q2A", 0.0) + scores_by_qid.get("Q2B", 0.0)
    q3_total = scores_by_qid.get("Q3A", 0.0) + scores_by_qid.get("Q3B", 0.0)
    
    best_choice = max(q1_total, q2_total)
    return round(best_choice + q3_total, 2)

if __name__ == "__main__":
    # Exact transcription from your uploaded SDMCET Question Paper
    sdmcet_qp = """
    SDM College of Engineering & Technology, Dharwad - 2
    Internal Assessment - I
    Note: Answer any one full question from Q.No.1 & Q.No.2.; Q.No.3 is COMPULSORY.
    Q. No. 1
    a) What is a distributed system? Discuss the goals that should be met to build an efficient distributed system. [5M, CO:1]
    b) Explain different types of transparency exist in distributed systems. [5M, CO:1]
    Q. No. 2
    a) Illustrate with an example, the use of Lamport's logical clocks to correct the clocks running at different rates. [5M, CO:2]
    b) What are the reasons for migrating code? Explain the various code migration models. [5M, CO:2]
    Q. No.3
    a) Explain the Network Time Protocol used in clock synchronization algorithms. [5M, CO:2]
    b) Explain the Bully election algorithms applied in the distributed systems. [5M, CO:2]
    """
    
    parsed = parse_sdmcet_question_paper(sdmcet_qp)
    print("--- Extracted SDMCET Questions ---")
    for q in parsed:
        print(f"{q['question_id']} ({q['max_marks']}M, {q['co']}): {q['question_text']}")