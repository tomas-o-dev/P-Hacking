# P-Hacking
### Comparing performance of binary classification models

For formal evaluation of the relative performance a set of classifiers, the null hypothesis is “the results from the models do not differ significantly”. The level of signiﬁcance is known as the p-value: the smaller the p-value, the stronger the evidence against the null hypothesis. The threshold significance level (usually called ‘alpha’) for a single statistical hypothesis test represents the probability of a Type I error: rejecting the null hypothesis when it is actually true. 

The standard method is to use an “omnibus” test to determine if the results from one or more of the models in the set differ significantly from the others (rejecting the null hypothesis), and then use a “post-hoc” test on each pair to find the significant differences.

Regarding post-hoc testing, Schaffer (1995) says:
>"In general, in testing any single hypothesis, conclusions based on statistical evidence are uncertain. We typically specify an acceptable maximum probability of rejecting the null hypothesis when it is true, thus committing a Type I error, and base the conclusion on the value of a statistic meeting this specification, preferably one with high power. When many hypotheses are tested, and each test has a specified Type I error probability, the probability that at least some Type I errors are committed increases, often sharply, with the number of hypotheses. This may have serious consequences if the set of conclusions must be evaluated as a whole. Numerous methods have been proposed for dealing with this problem, but no one solution will be acceptable for all situations." 

This library provides a convenient way to see the effect of the most common methods for proper multiple hypothesis testing, given a metric for the performance of each classifier (e.g., AUC, gmean, etc.) in a set. 
The term "p-hacking" is typically defined as a mix of strategies targeted at ensuring significant test results. While some of these strategies are considered less ethical than others (see Stefan et.al. (2023) for a review), choosing a metric for comparison and an appropriate correction  method for multiple hypothesis testing is simply a necessary step.  

For comparing the performance of binary classification models, the definitive papers are (Dietterich, 1998) and (Demšar, 2006). Based on their recomendations, two different evaluation strategies are available:

* Comparing multiple classifiers trained on _the same test set_:<br> 
Cochran’s Q as the omnibus test, and the McNemar test with correction for multiple hypothesis testing as the post hoc procedure.

* Comparing multiple classifiers trained on _multiple datasets_:<br> 
Friedman ranking test is used as the omnibus test, with the Nemenyi test as the post hoc procedure for all-to-all comparisons, or one of the general methods for one-to-all (control group) comparisons. 

#### Callable functions

* `run_friedman(indf, alpha=0.05)`<br>Wrapper for friedman_test() forked from the stac library: https://tec.citius.usc.es/stac/doc/
  - Parameters
    - `indf` : pd.Dataframe with the sample measurements for each group
    - `alpha` : float, significance threshold 
  - Returns
    - `sig` : boolean, Reject H0?   
    - `rptstr` : string, Accept or Reject H0 
    - `rankdic` : dict, "pivotal quantities" for the post-hoc tests

* `ph_pvals(ranks,control=None,nmyi=False,shaf=False)`<br> Post-Hoc p_values adjusted for multiple testing 
  - Parameters
    - `ranks` : dict, returned from Freidman test (rankdic)
    - `control` : string, groupname for one-to-all (control group) comparisons, default is all-vs-all comparison
    - `nmyi` : boolean, Run the Nemenyi test<br>Note: nemenyi_test is not appropriate for one-vs-all comparison
    - `shaf` : boolean, Run the Schaffer_static_test<br>Note: schaffer_static uses a recursive call; this causes internal python multithreading to go wildly oversubscribed when there are more than 18 classfiers to compare
  - Returns
    - pd.Dataframe with adjusted p_values from various methods
      - General: BDun, Sidak, Holm, Finner, Hochberg, Li
      - Nemenyi test: Nemenyi, BDun, Sidak (all single-step)
      - Schaffer test: Schaffer, Holm, Finner (all step-down)

* `cq_mph(y_test,clf_preds,cq=True,control=None,alpha=0.05)`<br>Cochrans_Q omnibus test and McNemar post-hoc test - requires mlxtend.evaluate
  - Parameters
    - `y_test` : array-like, shape=[n_samples], ground-truth class labels 
    - `clf_preds` : nested list, len=number of classifiers, where <br>clf_preds[n][0] : classifier name (string) <br>clf_preds[n][1] : array, shape=[n_samples], predicted class labels
    - `cq` : Boolean, Run the Cochrans_Q omnibus test
    - `control` : string, groupname for one-to-all (control group) comparisons, default is all-vs-all comparison
    - `alpha` : float, significance threshold 
  - Returns
    - pd.Dataframe with adjusted p_values (ph_pvals() General)

* `ap2h0(indf, alpha=0.05)`<br> Converts dataframe returned by ph_pvals() or cq_mph() from adjusted p_values to T/F for null hypothesis
  - Parameters
    - `indf` : Dataframe returned by ph_pvals() or cq_mph()
    - `alpha` : float, significance threshold 
  - Returns
    - pd.Dataframe with T/F for null hypothesis 

* `nemenyi_cdgraph(indf, alpha=0.05, pngpath=None, imgres=300)`<br> Forked from AutoRank: https://sherbold.github.io/autorank/ 
  - Parameters
    - `indf` : pd.Dataframe with the sample measurements for each group
    - `alpha` : float, significance threshold 
    - `pngpath` : string, where to save the file, default=None == [cwd]/cdgraf.png
    - `imgres` : int, dpi
  - Returns
    - `rptstr` : string, summary of significant differences
    - `cdplot` : string, filepath 

#### Library Requirements

* pandas
* numpy
* itertools
* scipy, scipy.stats 
* statsmodels.stats.libqsturng 
* mlxtend.evaluate
* sklearn.preprocessing

#### Examples
Jupyter notebooks in the `Demo` folder

#### Corrections for multiple hypothesis testing
The test functions return a Dataframe with adjusted p_values from various methods. A brief summary of their calculation:

> H0    : null hypothesis, difference between the two is not significant<br>
  `p`     = p_value for each pair being compared, from the normal distribution except Nemenyi (t-dist)<br>
  : unadjusted p_values are sorted such that p(0)<p(1)<br>
  `k`     = number of unadjusted p-values<br>
  `p(i)` = value at position (i) in the sorted list<br>
  `ap(i)` = value p(i) adjusted for multiple tests<br>

* In one-step adjustment methods p-values are compared to a
  predetermined value that is a function of the significance
  level (alpha), and the number of p-values (k).
  - Bonferroni-Dunn: `ap(i) = p(i)*k`
  - Sidak: `           ap(i) = 1-(1-p(i))^k`
  - Nemenyi: `        ap(i) = (p(i)*k*(k-1))/2`

* In step-down methods p-values are examined in order, from smallest
  to largest. Once a p-value is found that is large according to a
  criterion based on alpha and the position in the list, 
  H0 for that p-value and all larger p-values is accepted.
  - Holm: `   ap(i) = p(i)*(k-i+1)`
  - Finner: `  ap(i) = 1-(1-p(i))^(k/i)`  
  - Schaffer: `ap(i) = p(i)*t(i)`<br>
         where t(i) is the maximum number of hypotheses 
         which can be true given that any (i−1) hypotheses 
         are false; determined by a recursive function.

* In step-up methods p-values are examined in order, from largest to
  smallest. Once a p-value is found that is small according to a
  criterion based on alpha and the position in the list,
  H0 for that p-value and all smaller p-values is rejected.
  - Hochberg: `ap(i) = p(i)*(k-i+1)`
  - Li: `        ap(i) = p(i)/(p(i)+1-p(k))`

 #### References
 coming soon ...
 
