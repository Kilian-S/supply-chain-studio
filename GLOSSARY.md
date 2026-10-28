# Glossary: reading list

Terms used in the SKU clustering model that stay inside the "How it works" drawers until you have
read up on them. Each entry: the idea in one line, how it works, how we use it, and where to read
more. Figures are from the current run (world v1, seed 7; default model: GMM with k = 8).

---

## Gaussian mixture model (GMM)

**Idea.** Assume the data was produced by k overlapping "bell curves" (Gaussian distributions), one
per cluster, and find the curves that explain the data best.

**How it works.** The density of a feature vector x is a weighted sum of k Gaussians:

p(x) = Σ_c π_c · N(x | μ_c, Σ_c)

π_c is the share of SKUs in cluster c, μ_c its centre (centroid), Σ_c its spread. We use a
diagonal Σ_c: each feature has its own spread per cluster, with no correlations inside a cluster,
which keeps the number of parameters manageable for 300 SKUs and 16 features.

**Here.** The default clustering method. Unlike k-means it allows clusters of different sizes and
spreads, and gives each SKU a probability per cluster (see soft assignment).

**Read more.** scikit-learn user guide, Gaussian mixture models:
https://scikit-learn.org/stable/modules/mixture.html

---

## Soft assignment

**Idea.** Instead of putting each SKU in exactly one cluster (hard assignment, as k-means does),
give it a probability of belonging to each cluster.

**How it works.** For a fitted GMM, the probability that x belongs to cluster c (the posterior or
"responsibility") is π_c · N(x | μ_c, Σ_c) / Σ_j π_j · N(x | μ_j, Σ_j). The SKU is displayed in its
most likely cluster; the probability says how sure the model is.

**Here.** Borderline SKUs (probability well below 1) sit between two clusters; they are the ones a
planner should check by hand.

**Read more.** Same scikit-learn page as GMM; Bishop, *Pattern Recognition and Machine Learning*
(2006), chapter 9.

---

## Expectation maximisation (EM)

**Idea.** The algorithm that fits a GMM: guess the clusters, assign SKUs softly, re-estimate the
clusters from those assignments, repeat until nothing changes.

**How it works.** Two alternating steps. E-step: compute every SKU's soft assignment under the
current parameters. M-step: recompute π_c, μ_c and Σ_c as weighted averages using those
assignments. Each round increases the likelihood; it converges to a local optimum, so we start it
3 times from different points and keep the best.

**Here.** Used inside scikit-learn's `GaussianMixture`; you never call it directly.

**Read more.** Wikipedia, Expectation-maximisation algorithm:
https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm

---

## UMAP (Uniform Manifold Approximation and Projection)

**Idea.** A dimensionality reduction method that keeps each point's nearest neighbours close,
so clusters stay visible when 16 features are squeezed into 3 dimensions.

**How it works.** Builds a graph connecting each SKU to its nearest neighbours in the full feature
space, then finds a 3D layout whose neighbour graph looks as similar as possible. Local
neighbourhoods are kept well; distances between far-apart regions and the axes themselves carry no
meaning.

**Here.** The default layout of the 3D map (15 neighbours, minimum distance 0.15). This is why the
map has no axis labels.

**Read more.** UMAP documentation, How UMAP works:
https://umap-learn.readthedocs.io/en/latest/how_umap_works.html

---

## PCA (principal component analysis)

**Idea.** Rotate the data so the first axis follows the direction of largest spread, the second
the next largest at right angles, and so on; then keep the first few axes.

**How it works.** The axes (principal components) are the eigenvectors of the covariance matrix
of the standardised features. Each component is a weighted mix of the original features, and its
eigenvalue says how much of the total variance it carries.

**Here.** The alternative map layout. The first three components keep 60% of the variance (25%,
19% and 16%). Unlike UMAP, PCA axes do mean something, so they could be labelled by the features
that weigh most in each.

**Read more.** scikit-learn user guide, PCA:
https://scikit-learn.org/stable/modules/decomposition.html#pca

---

## Purity

**Idea.** For each cluster, find its most common true type; purity is the share of all SKUs that
belong to their cluster's most common type.

**How it works.** purity = (1/n) Σ_c max_t |cluster c ∩ type t|. It is 1 when every cluster holds
one type. It rises automatically with more clusters (with n clusters it is always 1), so it is only
fair between groupings of similar size.

**Here.** Our clusters (k = 8) reach 84%; ABC/XYZ analysis (9 classes) 73%; the demand classes by
ADI and CV² (4 classes) 48%.

**Read more.** Manning, Raghavan and Schütze, *Introduction to Information Retrieval*, section 16.3:
https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html

---

## ARI (adjusted Rand index)

**Idea.** Look at every pair of SKUs: do the clustering and the ground truth agree on whether the
two belong together? Then correct for how much agreement chance alone would give.

**How it works.** The Rand index is the share of pairs on which two groupings agree. The adjusted
version subtracts its expected value under random labelling and rescales, so 1 means identical
groupings, 0 means no better than chance (it can go slightly negative). It does not reward more
clusters the way purity does.

**Here.** Our clusters 0.42; ABC/XYZ analysis 0.33; demand classes 0.13. The best feature set
(without size and price) reaches 0.80, chosen after the fact (selection bias).

**Read more.** scikit-learn user guide, Rand index:
https://scikit-learn.org/stable/modules/clustering.html#rand-index

---

## NMI (normalised mutual information)

**Idea.** How much knowing a SKU's cluster tells you about its true type, scaled from 0 (nothing)
to 1 (everything).

**How it works.** Mutual information I(C; T) measures the information shared between cluster
labels C and true types T; dividing by the average of their entropies normalises it to 0 to 1.

**Here.** Our clusters 0.66; ABC/XYZ analysis 0.48; demand classes 0.34. A second check that agrees
with ARI on the ranking.

**Read more.** scikit-learn user guide, mutual information based scores:
https://scikit-learn.org/stable/modules/clustering.html#mutual-information-based-scores

---

## Silhouette score

**Idea.** For each SKU: is it much closer to the other members of its own cluster than to the
nearest other cluster?

**How it works.** s(i) = (b(i) − a(i)) / max(a(i), b(i)), where a(i) is the mean distance to its
own cluster and b(i) the mean distance to the nearest other cluster. Ranges from −1 to 1; the score
of a clustering is the mean over all SKUs.

**Here.** 0.12 for the default. Low values mean clusters overlap: SKUs form a continuum rather than
separate islands, which is normal for demand data and one reason we also check stability and the
ground truth.

**Read more.** scikit-learn user guide, silhouette coefficient:
https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient

---

## BIC (Bayesian information criterion)

**Idea.** A score for comparing models that rewards fitting the data well but penalises having many
parameters, so it does not always prefer more clusters.

**How it works.** BIC = −2 ln L̂ + p ln n, with L̂ the maximised likelihood, p the number of
parameters and n the number of SKUs. Lower is better.

**Here.** Chooses k in advance for the GMM: lowest at k = 8 (3,556), against 3,716 at k = 6 and
3,618 at k = 10.

**Read more.** Wikipedia, Bayesian information criterion:
https://en.wikipedia.org/wiki/Bayesian_information_criterion

---

## Bootstrap stability

**Idea.** If clusters are real, refitting on a slightly different sample of the data should give
almost the same clusters. If they change a lot, they are noise.

**How it works.** Draw a random 80% of the SKUs, refit, and compare the new labels with the original
ones on the SKUs both share (using ARI). Repeat 8 times and average.

**Here.** 0.76 ± 0.09 for the default: the clusters mostly survive resampling.

**Read more.** von Luxburg, "Clustering stability: an overview" (2010):
https://arxiv.org/abs/1007.1075

---

## Syntetos-Boylan classification

**Idea.** A standard way to classify demand patterns for forecasting and inventory, using two
numbers: how often demand occurs (ADI) and how much its size varies (CV²).

**How it works.** Cut-offs ADI = 1.32 and CV² = 0.49 give four classes: smooth (frequent, stable
size), erratic (frequent, variable size), intermittent (rare, stable size) and lumpy (rare,
variable size). Each class suggests a forecasting method (for example Croston's method for
intermittent demand).

**Here.** One of the baselines, shown as "demand classes by ADI and CV²": 167 smooth, 50 erratic,
52 intermittent, 31 lumpy SKUs.

**Read more.** Syntetos, Boylan and Croston, "On the categorization of demand patterns", *Journal
of the Operational Research Society* 56(5), 2005, pages 495 to 503.

---

## STL decomposition (Seasonal and Trend decomposition using Loess)

**Idea.** Split a time series into three parts: a slow trend, a repeating seasonal pattern, and the
remainder.

**How it works.** Alternates smoothing (loess, a local regression) to estimate the trend and the
seasonal component until they settle. Seasonal strength is then
F_S = max(0, 1 − Var(remainder) / Var(seasonal + remainder)): 0 for no yearly pattern, close to 1
for a strong one.

**Here.** Computes the "Seasonal peak" feature from each SKU's weekly sales with a 52-week period.

**Read more.** Cleveland et al., "STL: A seasonal-trend decomposition procedure based on loess",
*Journal of Official Statistics* 6(1), 1990; Hyndman and Athanasopoulos, *Forecasting: Principles
and Practice*, sections 3.6 and 4.3: https://otexts.com/fpp3/stl.html
