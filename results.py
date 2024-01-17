import pandas as pd
import matplotlib.pyplot as plt

# Data for Intra-topic and Inter-topic BERT Scores
intra_topic_data = {
    "Topic": ["Cod", "Horse", "Knife", "Shoe"],
    "BERT Score": [0.10585523582994938, 0.10850542248226702, 0.3062361652652423, 0.19640808552503586]
}
inter_topic_data = {
    "Topic Comparison": ["Cod vs Horse", "Cod vs Knife", "Cod vs Shoe", "Horse vs Knife", "Horse vs Shoe", "Knife vs Shoe"],
    "BERT Score": [0.30535903573036194, 0.3307549059391022, 0.1453770101070404, 0.3122301399707794, 0.20788560807704926, 0.21604400873184204]
}

# Convert to DataFrame
intra_topic_df = pd.DataFrame(intra_topic_data)
inter_topic_df = pd.DataFrame(inter_topic_data)

# Plotting function
def plot_table(df, title):
    fig, ax = plt.subplots(figsize=(8, 3))  # Adjust size as needed
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=df.columns, cellLoc = 'center', loc='center')
    plt.title(title)

    plt.show()

# Plot the tables
plot_table(intra_topic_df, "Intra-topic BERT Scores")
plot_table(inter_topic_df, "Inter-topic BERT Scores")
