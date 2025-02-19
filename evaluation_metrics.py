from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

class EvaluationMetrics:
    """
    A class that computes text-based evaluation metrics such as BLEU and ROUGE.
    """
    def __init__(self):
        # Initialize the ROUGE scorer
        self.rouge = Rouge()
    
    def bleu_score(self, reference: str, hypothesis: str) -> float:
        """
        Compute BLEU score between a reference and a hypothesis translation.
        """
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        # sentence_bleu expects a list of reference token lists
        score = sentence_bleu([ref_tokens], hyp_tokens)
        return score
    
    def rouge_scores(self, reference: str, hypothesis: str) -> dict:
        """
        Compute ROUGE scores between a reference and a hypothesis translation.
        Returns a dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores.
        """
        scores = self.rouge.get_scores(hypothesis, reference)
        return scores[0]
