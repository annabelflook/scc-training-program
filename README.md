# scc-training-program
For use in training programme development by unit training officers within the Sea Cadet Corps
No use data is included with this script for security reasons.

A collection of .csv files should be downloaded from Westminster and placed in a new directory called /TrainingData. This data will be imported into the code using extract_data()

```
df = main.extract_data()
```

A new Rank Class can be created by passing cadet ranks, and the syllabus, as arguments when instantiating the class.

```
cdt = Rank('Cdt', syllabus=['Cdt'])
```

Multiple ranks and each syllabus can be included per class. In this case, the algorithm will optimise for the best choice of modules.

```
cdt_cdt1 = Rank('Cdt', 'Cdt 1st', syllabus=['Cdt', 'Cdt 1st'])
```

Calling show_best_modules() for each rank will show the top suggested modules to encorporate into the training plan first.
By default, the top 5 modules will printed to the console, but this can be changed by passing in a new value for n. This function will also return the full list of modules and their corresponding 'scores'.

```
cdt.show_best_modules(df, n=10)  # This will show the top 10 best modules
```
