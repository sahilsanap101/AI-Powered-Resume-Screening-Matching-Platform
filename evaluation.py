def precision_at_k(ranked_labels, k=5):
    """
    ranked_labels: list of 1s and 0s sorted by model ranking
    """
    top_k = ranked_labels[:k]
    return sum(top_k) / k


def recall_at_k(ranked_labels, total_relevant, k=5):
    """
    total_relevant: total number of relevant resumes in dataset
    """
    top_k = ranked_labels[:k]
    return sum(top_k) / total_relevant if total_relevant > 0 else 0


# ---------------- EXAMPLE USAGE ---------------- #

# Model ranked resumes (Top → Bottom)
# 1 = relevant, 0 = not relevant
ranked_labels = [1, 1, 0, 1, 0, 0, 1]

total_relevant_resumes = 4

p_at_5 = precision_at_k(ranked_labels, k=5)
r_at_5 = recall_at_k(ranked_labels, total_relevant_resumes, k=5)

print("Precision@5:", round(p_at_5, 2))
print("Recall@5:", round(r_at_5, 2))