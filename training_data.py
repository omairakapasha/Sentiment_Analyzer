def get_sample_data():
    texts = [
        "I love this product, it is amazing",
        "This is the worst experience I have ever had",
        "Absolutely fantastic service",
        "I hate this so much",
        "Not good, very disappointing",
        "I am extremely happy with the results",
        "This is terrible and unacceptable",
        "I am not satisfied with the quality",
        "Great value for money",
        "Awful, would not recommend"
    ]

    labels = [
        "positive",
        "negative",
        "positive",
        "negative",
        "negative",
        "positive",
        "negative",
        "negative",
        "positive",
        "negative"
    ]

    return texts, labels
