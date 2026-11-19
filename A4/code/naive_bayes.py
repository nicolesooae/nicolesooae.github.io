"""
SYDE 572 Assignment 4: text classification with Naive Bayes.

    Required
      reading the data          : read_documents, split
      parameter estimation      : train_nb  (multinomial NB, additive smoothing, log probabilities)
      scoring / classifying     : score_doc_label, classify_nb, classify_documents, accuracy
      statistics                : wilson_interval (95% CI of an accuracy), mcnemar (exact p-value for two classifiers)
      error analysis            : most_confident_errors
    Optional
      six-category classifier   : train_nb on the topic labels
      learning curve            : learning_curve
      informative features      : top_features / print_top_nb
      numerical data            : train_gnb, score_gnb, classify_gnb (Gaussian NB on Iris)
      scikit-learn              : train_sklearn (TF-IDF + LinearSVC), the only place sklearn is used for classifying
      better features           : mark_negation, add_bigrams
      domain sensitivity        : cross_domain

    python naive_bayes.py      (prints every number in the report and saves the plots to ../media/)

The Naive Bayes code uses only the standard library and NumPy.
"""

import math
import os
from codecs import open
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(HERE, "..", "media")
DATA = os.path.join(HERE, "all_sentiment_shuffled.txt")
SEED = 0


# ---------- reading the data ----------
def read_documents(doc_file, label_column=1):
    """One review per line: topic, polarity, document id, tokens. label_column=1 gives polarity, 0 gives topic."""
    docs, labels = [], []
    with open(doc_file, encoding="utf-8") as f:
        for line in f:
            words = line.strip().split()
            docs.append(words[3:])
            labels.append(words[label_column])
    return docs, labels


def split(docs, labels, frac=0.80):
    """The handout's split: the file is already shuffled, so the first 80% train and the rest evaluate."""
    k = int(frac * len(docs))
    return docs[:k], labels[:k], docs[k:], labels[k:]


# ---------- Naive Bayes ----------
def train_nb(documents, labels, alpha=1.0):
    """Multinomial Naive Bayes with additive (Lidstone) smoothing.

        P(c)        = N_c / N                                       (fraction of documents with label c)
        P(w | c)    = (count(w, c) + alpha) / (n_c + alpha * (|V| + 1))

    count(w, c) is how often word w occurs in documents of class c and n_c is the total number of tokens in class c.
    The extra +1 in the vocabulary size reserves one slot for any word never seen in training, so an unknown word
    gets probability alpha / (n_c + alpha * (|V| + 1)) and the distribution still sums to one. Everything is stored
    as log probabilities. Labels are not hard-coded: any set of categories works.
    """
    classes = sorted(set(labels))
    counts = {c: Counter() for c in classes}
    n_docs = Counter(labels)
    for doc, c in zip(documents, labels):
        counts[c].update(doc)
    vocab = set()
    for c in classes:
        vocab.update(counts[c])
    V = len(vocab) + 1                                   # +1 for the unknown-word slot
    model = {"classes": classes, "alpha": alpha, "vocab": vocab, "counts": counts,
             "log_prior": {}, "log_p": {}, "log_unk": {}}
    for c in classes:
        n_c = sum(counts[c].values())
        denom = math.log(n_c + alpha * V)
        model["log_prior"][c] = math.log(n_docs[c] / len(labels))
        model["log_p"][c] = {w: math.log(counts[c][w] + alpha) - denom for w in vocab}
        model["log_unk"][c] = math.log(alpha) - denom
    return model


def score_doc_label(document, label, model):
    """log P(label, document) = log P(label) + sum over tokens of log P(word | label)   (bag of words, independence)."""
    log_p, unk = model["log_p"][label], model["log_unk"][label]
    return model["log_prior"][label] + sum(log_p.get(w, unk) for w in document)


def classify_nb(document, model):
    """The label with the highest joint log probability (same as the highest posterior, since P(document) is shared)."""
    return max(model["classes"], key=lambda c: score_doc_label(document, c, model))


def classify_documents(docs, model):
    return [classify_nb(d, model) for d in docs]


def accuracy(true_labels, guessed_labels):
    return sum(t == g for t, g in zip(true_labels, guessed_labels)) / len(true_labels)


def posterior(document, model):
    """P(c | document) for every class, computed stably with log-sum-exp."""
    s = np.array([score_doc_label(document, c, model) for c in model["classes"]])
    p = np.exp(s - s.max())
    return dict(zip(model["classes"], p / p.sum()))


# ---------- statistics ----------
def wilson_interval(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n (here: an accuracy on n test documents)."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def mcnemar(true, pred_a, pred_b):
    """Exact (binomial) McNemar test on the same test set. b = A right / B wrong, c = A wrong / B right.
    Under H0 (equal accuracy) the discordant documents split 50/50; two-sided p-value."""
    b = sum(a == t != p for t, a, p in zip(true, pred_a, pred_b))
    c = sum(p == t != a for t, a, p in zip(true, pred_a, pred_b))
    n, k = b + c, min(b, c)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return b, c, p


def report(name, true, pred):
    """Accuracy with its 95% CI, as a printed line and as a tuple."""
    k, n = sum(t == g for t, g in zip(true, pred)), len(true)
    lo, hi = wilson_interval(k, n)
    print(f"    {name:42s} {100 * k / n:6.2f}%   95% CI [{100 * lo:.2f}, {100 * hi:.2f}]   ({k}/{n})")
    return k / n, lo, hi


# ---------- error analysis ----------
def most_confident_errors(docs, labels, model, max_len=None, n=5):
    """Misclassified documents sorted by how sure the model was: the log-odds margin of the wrong label over the right one."""
    errs = []
    for i, (d, t) in enumerate(zip(docs, labels)):
        if max_len and len(d) > max_len:
            continue
        g = classify_nb(d, model)
        if g != t:
            errs.append((score_doc_label(d, g, model) - score_doc_label(d, t, model), i, t, g))
    return sorted(errs, reverse=True)[:n]


def word_contributions(document, model, a="pos", b="neg", n=6):
    """The words that pushed hardest towards a and towards b: log P(w|a) - log P(w|b), summed over occurrences."""
    contrib = Counter()
    for w in document:
        contrib[w] += model["log_p"][a].get(w, model["log_unk"][a]) - model["log_p"][b].get(w, model["log_unk"][b])
    ranked = contrib.most_common()
    return ranked[:n], ranked[::-1][:n]


# ---------- optional: smoothing and learning curve ----------
def tune_alpha(docs, labels, alphas, folds=5):
    """k-fold CV on the training data only, so the test set is not used to choose alpha."""
    idx = np.arange(len(docs)) % folds
    out = []
    for a in alphas:
        accs = []
        for f in range(folds):
            tr = [i for i in range(len(docs)) if idx[i] != f]
            te = [i for i in range(len(docs)) if idx[i] == f]
            m = train_nb([docs[i] for i in tr], [labels[i] for i in tr], alpha=a)
            accs.append(accuracy([labels[i] for i in te], classify_documents([docs[i] for i in te], m)))
        out.append((a, float(np.mean(accs))))
    return out


def learning_curve(train_docs, train_labels, test_docs, test_labels, fracs, trainers):
    """Accuracy on the fixed 20% test set when training on the first fraction of the 80% training part."""
    rows = []
    for f in fracs:
        k = int(round(f * len(train_docs)))
        row = [f, k]
        for train, classify in trainers:
            pred = classify(test_docs, train(train_docs[:k], train_labels[:k]))
            hits = sum(t == g for t, g in zip(test_labels, pred))
            row.append((hits / len(test_labels),) + wilson_interval(hits, len(test_labels)))
        rows.append(row)
    return rows


# ---------- optional: informative features ----------
def top_features(model, k=15, a="pos", b="neg", min_count=0):
    """Rank words by the log-likelihood ratio log P(w|a) - log P(w|b). A large alpha pulls rare words towards 0,
    so the top of the list is words that are both frequent and lopsided rather than one-off typos."""
    vocab = [w for w in model["vocab"] if model["counts"][a][w] + model["counts"][b][w] >= min_count]
    llr = {w: model["log_p"][a][w] - model["log_p"][b][w] for w in vocab}
    top_a = sorted(llr, key=lambda w: (-llr[w], w))[:k]          # ties (common for rare words) broken alphabetically
    top_b = sorted(llr, key=lambda w: (llr[w], w))[:k]
    return [(w, llr[w]) for w in top_a], [(w, llr[w]) for w in top_b]


def print_top_nb(k, model, a="pos", b="neg"):
    pos, neg = top_features(model, k, a, b)
    show = lambda w: w.encode("unicode_escape").decode()        # the data contains a few tokens with control characters
    print(f"    {a:>8s}: " + ", ".join(show(w) for w, _ in pos))
    print(f"    {b:>8s}: " + ", ".join(show(w) for w, _ in neg))


def top_by_topic(model, k=8):
    """One-vs-rest for the topic classifier: log P(w|topic) - mean over the other topics of log P(w|other)."""
    out = {}
    for c in model["classes"]:
        others = [o for o in model["classes"] if o != c]
        llr = {w: model["log_p"][c][w] - np.mean([model["log_p"][o][w] for o in others]) for w in model["vocab"]}
        out[c] = sorted(llr, key=lambda w: (-llr[w], w))[:k]
    return out


# ---------- optional: better features ----------
NEGATIONS = {"not", "no", "never", "n't", "nothing", "nobody", "none", "nor", "neither", "cannot", "without"}
CLAUSE_END = {".", ",", "!", "?", ";", ":", "(", ")", "but"}


def mark_negation(doc):
    """Pang et al. (2002): prefix NOT_ to every token after a negation word until the next punctuation mark."""
    out, neg = [], False
    for w in doc:
        if w in CLAUSE_END:
            neg = False
            out.append(w)
        elif w in NEGATIONS:
            neg = True
            out.append(w)
        else:
            out.append("NOT_" + w if neg else w)
    return out


def add_bigrams(doc):
    """Unigrams plus adjacent word pairs, e.g. 'not good' becomes a feature of its own."""
    return doc + [a + " " + b for a, b in zip(doc, doc[1:])]


# ---------- optional: domain sensitivity ----------
def cross_domain(docs, topics, sentiment, alpha=1.0):
    """Split each topic 80/20, train on one topic's 80%, test on every topic's 20%. Returns a len(topics)^2 matrix."""
    parts = {}
    for t in sorted(set(topics)):
        d = [doc for doc, tt in zip(docs, topics) if tt == t]
        y = [s for s, tt in zip(sentiment, topics) if tt == t]
        parts[t] = split(d, y)
    names = sorted(parts)
    M = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        m = train_nb(parts[a][0], parts[a][1], alpha)
        for j, b in enumerate(names):
            M[i, j] = accuracy(parts[b][3], classify_documents(parts[b][2], m))
    return names, M, parts


# ---------- optional: Naive Bayes for numerical data ----------
def load_iris():
    rows = [line.strip().split(",") for line in open(os.path.join(HERE, "iris.data"), encoding="utf-8") if line.strip()]
    X = np.array([[float(v) for v in r[:4]] for r in rows])
    Y = [r[4].replace("Iris-", "") for r in rows]
    return X, Y


def train_gnb(X, Y):
    """Gaussian NB: for each class, the prior and the ML mean and variance of each column (12 Gaussians for Iris)."""
    Y = np.asarray(Y)
    classes = sorted(set(Y))
    return {"classes": classes,
            "log_prior": {c: math.log(np.mean(Y == c)) for c in classes},
            "mean": {c: X[Y == c].mean(axis=0) for c in classes},
            "var": {c: X[Y == c].var(axis=0) for c in classes}}          # ML estimate (divide by N)


def score_gnb(x, label, model):
    """log P(label) + sum over columns of the log of the normal pdf: the word log probabilities become log densities."""
    mu, var = model["mean"][label], model["var"][label]
    return model["log_prior"][label] + float(np.sum(-0.5 * np.log(2 * np.pi * var) - (x - mu) ** 2 / (2 * var)))


def classify_gnb(X, model):
    return [max(model["classes"], key=lambda c: score_gnb(x, c, model)) for x in X]


# ---------- optional: scikit-learn ----------
def train_sklearn(docs, labels):
    """The handout's pipeline: TF-IDF features, SelectKBest(all), linear SVM."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC
    vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, token_pattern=None)
    return make_pipeline(vec, SelectKBest(k="all"), LinearSVC()).fit(docs, labels)


def classify_documents_sklearn(docs, model):
    return list(model.predict(docs))


# ---------- driver ----------
if __name__ == "__main__":
    import plots

    os.makedirs(MEDIA, exist_ok=True)
    plots.setup_style()
    DOCS, LABELS = read_documents(DATA)
    TOPICS = read_documents(DATA, label_column=0)[1]

    train_docs, train_labels, eval_docs, eval_labels = split(DOCS, LABELS)
    print(f"Data: {len(DOCS)} reviews, {Counter(LABELS)}")
    print(f"    train {len(train_docs)} ({Counter(train_labels)}),  test {len(eval_docs)} ({Counter(eval_labels)})")
    print(f"    tokens per review: median {int(np.median([len(d) for d in DOCS]))}, "
          f"min {min(len(d) for d in DOCS)}, max {max(len(d) for d in DOCS)}")

    # ----- training -----
    model = train_nb(train_docs, train_labels, alpha=1.0)
    print(f"\nModel (alpha = 1): |V| = {len(model['vocab'])},  tokens per class "
          + ", ".join(f"{c} {sum(model['counts'][c].values())}" for c in model["classes"]))
    print("    log P(c) = " + ", ".join(f"{c} {model['log_prior'][c]:.4f} (P = {math.exp(model['log_prior'][c]):.4f})"
                                    for c in model["classes"]))

    # ----- sanity checks -----
    print("\nSanity check 1: one-word documents, P(label, doc)")
    for w in ["great", "bad", "excellent", "waste", "the", "boring"]:
        p = {c: math.exp(score_doc_label([w], c, model)) for c in model["classes"]}
        print(f"    [{w!r}]  pos {p['pos']:.6f}   neg {p['neg']:.6f}   -> {classify_nb([w], model)}"
              f"   (counts: pos {model['counts']['pos'][w]}, neg {model['counts']['neg'][w]})")
    doc = ["a", "top-quality", "performance"]
    print(f"Sanity check 2: {doc}  in vocabulary: {[w in model['vocab'] for w in doc]}")
    print("    " + ",  ".join(f"log P({c}, doc) = {score_doc_label(doc, c, model):.3f}" for c in model["classes"])
          + f"   -> {classify_nb(doc, model)}")
    print("Small documents:")
    small = [["this", "is", "not", "good"], ["not", "bad", "at", "all"], ["i", "loved", "it"],
             ["a", "waste", "of", "money"], ["highly", "recommend"], ["returned", "it"]]
    for d in small:
        print(f"    {' '.join(d):22s} -> {classify_nb(d, model)}   P(pos|doc) = {posterior(d, model)['pos']:.3f}")

    # ----- evaluation -----
    print("\nEvaluation on the 20% test set")
    pred = classify_documents(eval_docs, model)
    report("Naive Bayes, alpha = 1", eval_labels, pred)
    tp = sum(t == g == "pos" for t, g in zip(eval_labels, pred))
    tn = sum(t == g == "neg" for t, g in zip(eval_labels, pred))
    fp = sum(t == "neg" and g == "pos" for t, g in zip(eval_labels, pred))
    fn = sum(t == "pos" and g == "neg" for t, g in zip(eval_labels, pred))
    print(f"    confusion (true x guess): pos/pos {tp}, pos/neg {fn}, neg/pos {fp}, neg/neg {tn}")
    print(f"    majority-class baseline: {100 * max(Counter(eval_labels).values()) / len(eval_labels):.2f}%")
    plots.plot_confusion([[tn, fp], [fn, tp]], ["neg", "pos"], "Sentiment, Naive Bayes",
                         os.path.join(MEDIA, "confusion_sentiment.png"))

    # smoothing parameter, chosen on the training data only
    alphas = [0.01, 0.03, 0.1, 0.3, 0.5, 1, 2, 3, 5, 10, 30]
    cv = tune_alpha(train_docs, train_labels, alphas)
    test_alpha = [(a, accuracy(eval_labels, classify_documents(eval_docs, train_nb(train_docs, train_labels, a))))
                  for a in alphas]
    print("    alpha:   " + "  ".join(f"{a:>6}" for a in alphas))
    print("    5-fold CV" + "".join(f"{100 * x:8.2f}" for _, x in cv))
    print("    test     " + "".join(f"{100 * x:8.2f}" for _, x in test_alpha))
    best_alpha = max(cv, key=lambda r: r[1])[0]
    print(f"    best alpha by CV on the training set: {best_alpha}")
    plots.plot_alpha(cv, test_alpha, os.path.join(MEDIA, "alpha.png"))
    model_best = train_nb(train_docs, train_labels, best_alpha)
    pred_best = classify_documents(eval_docs, model_best)
    report(f"Naive Bayes, alpha = {best_alpha}", eval_labels, pred_best)
    print(f"    McNemar alpha=1 vs alpha={best_alpha}: b, c, p = {mcnemar(eval_labels, pred, pred_best)}")

    # ----- error analysis -----
    print("\nError analysis: most confident mistakes among short reviews (<= 40 tokens)")
    for margin, i, t, g in most_confident_errors(eval_docs, eval_labels, model, max_len=40, n=8):
        d = eval_docs[i]
        toward_g, toward_t = word_contributions(d, model, g, t, n=5)
        print(f"  #{i}  true {t}, guessed {g}, log-odds {margin:.2f}, {len(d)} tokens, topic {TOPICS[len(train_docs) + i]}")
        print(f"      {' '.join(d)}")
        print(f"      towards {g}: " + ", ".join(f"{w} {v:+.2f}" for w, v in toward_g))
        print(f"      towards {t}: " + ", ".join(f"{w} {-v:+.2f}" for w, v in toward_t))
    lens = np.array([len(d) for d in eval_docs])
    wrong = np.array([t != g for t, g in zip(eval_labels, pred)])
    print("    error rate by review length:")
    for lo, hi in [(0, 50), (50, 100), (100, 200), (200, 400), (400, 10 ** 6)]:
        sel = (lens >= lo) & (lens < hi)
        print(f"      {lo:4d}-{hi if hi < 10 ** 6 else '':>4}: {100 * wrong[sel].mean():5.2f}%  (n = {sel.sum()})")
    margins = np.array([abs(score_doc_label(d, "pos", model) - score_doc_label(d, "neg", model)) for d in eval_docs])
    print("    error rate by confidence |log-odds|:")
    for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)]:
        sel = (margins >= lo) & (margins < hi)
        print(f"      {lo:3d}-{hi if hi < 1e9 else '':>3}: {100 * wrong[sel].mean():5.2f}%  (n = {sel.sum()})")
    plots.plot_margins(margins, wrong, lens, os.path.join(MEDIA, "errors.png"))

    # ===== optional tasks =====
    # ----- six-category topic classifier -----
    print("\nOptional: topic classifier (6 classes)")
    tr_d, tr_t, ev_d, ev_t = split(DOCS, TOPICS)
    topic_model = train_nb(tr_d, tr_t, alpha=1.0)
    pred_t = classify_documents(ev_d, topic_model)
    report("Naive Bayes, topic", ev_t, pred_t)
    names = topic_model["classes"]
    C = np.array([[sum(t == a and g == b for t, g in zip(ev_t, pred_t)) for b in names] for a in names])
    for a, row in zip(names, C):
        print(f"      {a:9s} " + " ".join(f"{v:4d}" for v in row) + f"   recall {100 * row[names.index(a)] / row.sum():.1f}%")
    plots.plot_confusion(C, names, "Topic, Naive Bayes", os.path.join(MEDIA, "confusion_topic.png"))
    print("    most typical words per topic:")
    by_topic = top_by_topic(train_nb(tr_d, tr_t, alpha=10.0), k=8)
    for c, ws in by_topic.items():
        print(f"      {c:9s} {', '.join(ws)}")

    # ----- learning curve -----
    print("\nOptional: learning curve (test set fixed, first f of the training part)")
    fracs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    lc = learning_curve(train_docs, train_labels, eval_docs, eval_labels, fracs,
                        [(train_nb, classify_documents), (train_sklearn, classify_documents_sklearn)])
    for f, k, nb, svm in lc:
        print(f"    {100 * f:5.0f}%  n = {k:5d}   NB {100 * nb[0]:6.2f}%   LinearSVC {100 * svm[0]:6.2f}%")
    plots.plot_learning_curve(lc, os.path.join(MEDIA, "learning_curve.png"))

    # ----- informative features -----
    print("\nOptional: most informative features (log P(w|pos) - log P(w|neg))")
    for a in [1.0, 10.0, 100.0]:
        print(f"  alpha = {a}")
        print_top_nb(15, train_nb(train_docs, train_labels, a))
    feat_model = train_nb(train_docs, train_labels, alpha=100.0)
    plots.plot_features(feat_model, top_features(feat_model, 20), os.path.join(MEDIA, "features.png"))

    # ----- scikit-learn -----
    print("\nOptional: scikit-learn TF-IDF + LinearSVC")
    svm = train_sklearn(train_docs, train_labels)
    pred_svm = classify_documents_sklearn(eval_docs, svm)
    report("LinearSVC (TF-IDF)", eval_labels, pred_svm)
    print(f"    McNemar NB vs SVM: b (NB right only), c (SVM right only), p = {mcnemar(eval_labels, pred, pred_svm)}")
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import CountVectorizer
    cvec = CountVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, token_pattern=None)
    sk_nb = MultinomialNB(alpha=1.0).fit(cvec.fit_transform(train_docs), train_labels)
    pred_sknb = list(sk_nb.predict(cvec.transform(eval_docs)))
    agree = np.mean([a == b for a, b in zip(pred, pred_sknb)])
    report("sklearn MultinomialNB (check)", eval_labels, pred_sknb)
    print(f"    agreement with my NB: {100 * agree:.2f}% of test documents")

    # ----- better features -----
    print("\nOptional: improved features (alpha = 1)")
    variants = {"unigrams (baseline)": lambda d: d,
                "negation marking": mark_negation,
                "unigrams + bigrams": add_bigrams,
                "negation + bigrams": lambda d: add_bigrams(mark_negation(d))}
    for name, fx in variants.items():
        m = train_nb([fx(d) for d in train_docs], train_labels, 1.0)
        p = classify_documents([fx(d) for d in eval_docs], m)
        report(name, eval_labels, p)
        b, c, pv = mcnemar(eval_labels, pred, p)
        print(f"        vs baseline: only baseline right {b}, only this right {c}, McNemar p = {pv:.2g}")
        if name == "negation + bigrams":
            print(f"        vs LinearSVC: (NB right only, SVM right only, p) = {mcnemar(eval_labels, p, pred_svm)}")
    print(f"    example: {' '.join(mark_negation('it is not very good , but cheap'.split()))}")

    # ----- domain sensitivity -----
    print("\nOptional: domain sensitivity (rows = trained on, columns = tested on)")
    dnames, M, parts = cross_domain(DOCS, TOPICS, LABELS)
    print("              " + "".join(f"{n:>9s}" for n in dnames))
    for n, row in zip(dnames, M):
        print(f"    {n:9s} " + "".join(f"{100 * v:9.1f}" for v in row))
    i, j = dnames.index("camera"), dnames.index("books")
    for a, b in [(i, j), (j, i)]:
        hits_in = round(M[a, a] * len(parts[dnames[a]][3]))
        hits_out = round(M[a, b] * len(parts[dnames[b]][3]))
        print(f"    {dnames[a]} model: in-domain {100 * M[a, a]:.1f}% "
              f"{tuple(round(100 * x, 1) for x in wilson_interval(hits_in, len(parts[dnames[a]][3])))}, "
              f"on {dnames[b]} {100 * M[a, b]:.1f}% "
              f"{tuple(round(100 * x, 1) for x in wilson_interval(hits_out, len(parts[dnames[b]][3])))}")
    for b in [j, i]:
        m_in = train_nb(parts[dnames[b]][0], parts[dnames[b]][1])
        m_out = train_nb(parts[dnames[i if b == j else j]][0], parts[dnames[i if b == j else j]][1])
        te_d, te_y = parts[dnames[b]][2], parts[dnames[b]][3]
        print(f"    McNemar on {dnames[b]} test set, in-domain vs out-of-domain model: "
              f"{mcnemar(te_y, classify_documents(te_d, m_in), classify_documents(te_d, m_out))}")
    off = M[~np.eye(len(M), dtype=bool)]
    print(f"    mean in-domain {100 * np.diag(M).mean():.1f}%, mean out-of-domain {100 * off.mean():.1f}%")
    plots.plot_domain(dnames, M, os.path.join(MEDIA, "domain.png"))

    # ----- Gaussian NB on Iris -----
    print("\nOptional: Gaussian Naive Bayes on Iris")
    X, Y = load_iris()
    rng = np.random.default_rng(SEED)
    eval_part = np.zeros(len(Y), bool)
    eval_part[rng.choice(len(Y), size=40, replace=False)] = True
    Yarr = np.array(Y)
    gnb = train_gnb(X[~eval_part], Yarr[~eval_part])
    for c in gnb["classes"]:
        print(f"    {c:10s} mean {np.round(gnb['mean'][c], 3)}  sd {np.round(np.sqrt(gnb['var'][c]), 3)}")
    pg = classify_gnb(X[eval_part], gnb)
    report("my Gaussian NB, 40 random test flowers", list(Yarr[eval_part]), pg)
    from sklearn.naive_bayes import GaussianNB
    psk = GaussianNB(var_smoothing=0).fit(X[~eval_part], Yarr[~eval_part]).predict(X[eval_part])
    print(f"    sklearn GaussianNB agrees on {np.mean(np.array(pg) == psk) * 100:.0f}% of test flowers")
    folds = np.array_split(rng.permutation(len(Y)), 10)
    cv_hits = 0
    for f in folds:
        mask = np.zeros(len(Y), bool)
        mask[f] = True
        cv_hits += sum(np.array(classify_gnb(X[mask], train_gnb(X[~mask], Yarr[~mask]))) == Yarr[mask])
    lo, hi = wilson_interval(cv_hits, len(Y))
    print(f"    10-fold CV: {100 * cv_hits / len(Y):.2f}%  95% CI [{100 * lo:.2f}, {100 * hi:.2f}]  ({cv_hits}/{len(Y)})")
    plots.plot_gnb(X, Yarr, gnb, os.path.join(MEDIA, "iris_gnb.png"))
