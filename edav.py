import nltk
from nltk.corpus import movie_reviews
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Ensure the movie_reviews corpus is downloaded
nltk.download('movie_reviews')

# === CONFIGURATION ===

# General plot saving options
save_pdf_plots     = True                       # Master switch to save plots
output_folder      = "plots"                    # Directory to save all .svg plots

# PDF plot options
save_combined_plot = True                       # Save combined PDF plot
combined_filename  = "pdf_combined_reviews.svg"  # Filename for combined PDF plot

# Individual plot saving toggles
save_topwords      = True                       # Save top words bar plots
save_wordcloud     = True                       # Save word cloud plots



# Create folder if needed
if save_pdf_plots and not os.path.exists(output_folder):
    os.makedirs(output_folder)

# === LOAD DATA ===
file_ids = movie_reviews.fileids()
categories = [movie_reviews.categories(fileid)[0] for fileid in file_ids]
word_counts = [len(movie_reviews.words(fileid)) for fileid in file_ids]

# Convert to NumPy for filtering
word_counts_np = np.array(word_counts)
categories_np = np.array(categories)

# Slices
all_reviews = word_counts_np
pos_reviews = word_counts_np[categories_np == "pos"]
neg_reviews = word_counts_np[categories_np == "neg"]

# === PLOT FUNCTION ===
def plot_density(data, label, color, filename=None):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data, fill=True, color=color, label=label)
    plt.title(f"Probability Density Function - {label}")
    plt.xlabel("Number of Words")
    plt.ylabel("Density")
    plt.legend()
    if save_pdf_plots and filename:
        plt.savefig(os.path.join(output_folder, filename), format="svg")
    plt.show()

# === PLOTS of PDFs Separate ===
plot_density(all_reviews, "All Reviews", "gray", filename="pdf_all_reviews.svg")
plot_density(pos_reviews, "Positive Reviews", "green", filename="pdf_positive_reviews.svg")
plot_density(neg_reviews, "Negative Reviews", "red", filename="pdf_negative_reviews.svg")




# === PLOT ALL PDFs TOGETHER ===
plt.figure(figsize=(12, 6))
sns.kdeplot(all_reviews, fill=True, color="gray", label="All Reviews", linewidth=2, alpha=0.4)
sns.kdeplot(pos_reviews, fill=True, color="green", label="Positive Reviews", linewidth=2, alpha=0.5)
sns.kdeplot(neg_reviews, fill=True, color="red", label="Negative Reviews", linewidth=2, alpha=0.5)

plt.title("Probability Density Function of Word Counts in Reviews")
plt.xlabel("Number of Words")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

# Save if needed
if save_combined_plot:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(os.path.join(output_folder, combined_filename), format="svg")

plt.show()


# Review Length Stats

from scipy.stats import describe

def summarize_lengths(lengths, label):
    summary = describe(lengths)
    print(f"\n{label} Review Length Stats:")
    print(f"Min: {summary.minmax[0]}, Max: {summary.minmax[1]}")
    print(f"Mean: {summary.mean:.2f}, Variance: {summary.variance:.2f}")
    print(f"Skewness: {summary.skewness:.2f}, Kurtosis: {summary.kurtosis:.2f}")

summarize_lengths(all_reviews, "All")
summarize_lengths(pos_reviews, "Positive")
summarize_lengths(neg_reviews, "Negative")

# Most Common Words (w/ Stopwords Removed)

from nltk.corpus import stopwords
from collections import Counter
import string
nltk.download('stopwords')

stop_words = set(stopwords.words("english"))
punct = set(string.punctuation)

def get_top_words(category, n=20):
    if category == "all":
        words = movie_reviews.words()
    else:
        words = movie_reviews.words(categories=[category])
    words = [w.lower() for w in words if w.lower() not in stop_words and w not in punct]
    return Counter(words).most_common(n)

# Plotting helper
def plot_top_words(word_freq, label, color, save=False, save_folder="plots"):
    words, freqs = zip(*word_freq)
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=list(freqs), y=list(words), hue=list(words), dodge=False, palette=color, legend=False)
    plt.title(f"Top Words in {label} Reviews")
    plt.xlabel("Frequency")
    plt.ylabel("Words")
    plt.tight_layout()

    if save:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(os.path.join(save_folder, f"top_words_{label.lower()}.svg"), format="svg")
    plt.show()


plot_top_words(get_top_words("all"), label="All", color="Blues_r", save=save_topwords, save_folder=output_folder)
plot_top_words(get_top_words("pos"), label="Positive", color="Greens_r", save=save_topwords, save_folder=output_folder)
plot_top_words(get_top_words("neg"), label="Negative", color="Reds_r", save=save_topwords, save_folder=output_folder)

# Word Cloud Visualization

from wordcloud import WordCloud

def generate_wordcloud(category, color, save=False, save_folder="plots"):
    words = movie_reviews.words(categories=[category])
    words = [w.lower() for w in words if w.lower() not in stop_words and w not in punct]
    text = " ".join(words)

    wc = WordCloud(width=800, height=400, background_color="white", colormap=color).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud - {category.title()} Reviews")
    plt.tight_layout()

    if save:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(os.path.join(save_folder, f"wordcloud_{category.lower()}.svg"), format="svg")
    plt.show()



generate_wordcloud("pos", color="Greens", save=save_wordcloud, save_folder=output_folder)
generate_wordcloud("neg", color="Reds", save=save_wordcloud, save_folder=output_folder)
